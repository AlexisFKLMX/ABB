import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from collections import Counter

# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Definir las emociones de AffectNet
EMOTIONS = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']
NUM_CLASSES = len(EMOTIONS)

# Hiperparámetros optimizados para AffectNet
BATCH_SIZE = 64  # Aumentado por el tamaño del dataset
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
IMAGE_SIZE = 224  # Tamaño estándar para modelos preentrenados
PATIENCE = 5  # Para early stopping

# Dataset personalizado para AffectNet en formato YOLO
class AffectNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, subset='train'):
        """
        Args:
            root_dir (string): Directorio raíz del dataset
            transform (callable, optional): Transformaciones opcionales
            subset (string): 'train', 'val' o 'test'
        """
        self.root_dir = root_dir
        self.transform = transform
        self.subset = subset
        self.images = []
        self.labels = []
        
        # Ruta a las imágenes y etiquetas
        image_dir = os.path.join(root_dir, subset, 'images')
        label_dir = os.path.join(root_dir, subset, 'labels')
        
        print(f"Cargando dataset {subset} desde {image_dir}...")
        
        # Cargar todas las imágenes y sus etiquetas
        if os.path.exists(image_dir) and os.path.exists(label_dir):
            for img_file in tqdm(os.listdir(image_dir), desc=f'Cargando {subset}'):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(image_dir, img_file)
                    label_file = os.path.splitext(img_file)[0] + '.txt'
                    label_path = os.path.join(label_dir, label_file)
                    
                    if os.path.exists(label_path):
                        with open(label_path, 'r') as f:
                            line = f.readline().strip()
                            if line:
                                label = int(line.split()[0])
                                if 0 <= label < NUM_CLASSES:  # Validar etiqueta
                                    self.images.append(img_path)
                                    self.labels.append(label)
        
        print(f"Total de imágenes en {subset}: {len(self.images)}")
        
        # Calcular distribución de clases
        self.class_counts = Counter(self.labels)
        print(f"Distribución de clases en {subset}:")
        for emotion_idx, count in sorted(self.class_counts.items()):
            print(f"  {EMOTIONS[emotion_idx]}: {count} imágenes")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # Si hay error al cargar la imagen, devolver una imagen negra
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Arquitectura de red neuronal optimizada para AffectNet
class AffectNetCNN(nn.Module):
    def __init__(self, num_classes=8, model_name='efficientnet'):
        super(AffectNetCNN, self).__init__()
        
        if model_name == 'efficientnet':
            # EfficientNet-B0 (recomendado para balance precisión/velocidad)
            self.base_model = models.efficientnet_b0(pretrained=True)
            in_features = self.base_model.classifier[1].in_features
            
            # Reemplazar clasificador
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
        
        elif model_name == 'resnet50':
            # ResNet50 (más preciso pero más pesado)
            self.base_model = models.resnet50(pretrained=True)
            
            # Congelar primeras capas
            for param in list(self.base_model.parameters())[:-20]:
                param.requires_grad = False
            
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
    
    def forward(self, x):
        return self.base_model(x)

# Transformaciones de datos optimizadas para AffectNet
def get_transforms():
    # Transformaciones para entrenamiento con más augmentación
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transformaciones para validación/test
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# Función para manejar el desbalance de clases
def get_weighted_sampler(dataset):
    class_counts = np.array([dataset.class_counts[i] for i in range(NUM_CLASSES)])
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in dataset.labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )

# Función de entrenamiento mejorada con early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Fase de entrenamiento
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Actualizar barra de progreso
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 
                                     'acc': f'{100 * correct / total:.2f}%'})
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Fase de validación
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        class_correct = list(0. for i in range(NUM_CLASSES))
        class_total = list(0. for i in range(NUM_CLASSES))
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Accuracy por clase
                c = (predicted == labels).squeeze()
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Ajustar learning rate
        scheduler.step()
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Imprimir accuracy por clase
        print('\nAccuracy por emoción:')
        for i in range(NUM_CLASSES):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                print(f'{EMOTIONS[i]}: {acc:.2f}%')
        
        print('-' * 80)
        
        # Early stopping y guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, 'best_affectnet_model.pth')
            print(f'Mejor modelo guardado con accuracy: {best_val_acc:.2f}%')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'Early stopping en epoch {epoch+1}')
                break
    
    return train_losses, val_losses, train_accs, val_accs

# Función mejorada para prueba con webcam
def test_webcam(model_path='best_affectnet_model.pth'):
    # Cargar modelo
    model = AffectNetCNN(num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Cargar clasificador de rostros
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Transformación para la imagen
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Colores para cada emoción
    emotion_colors = {
        'neutral': (128, 128, 128),
        'happy': (0, 255, 0),
        'sad': (255, 0, 0),
        'surprise': (0, 255, 255),
        'fear': (255, 0, 255),
        'disgust': (0, 128, 0),
        'anger': (0, 0, 255),
        'contempt': (255, 128, 0)
    }
    
    # Iniciar captura de video
    cap = cv2.VideoCapture(0)
    
    print("Presiona 'q' para salir, 's' para guardar screenshot")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Voltear horizontalmente para efecto espejo
        frame = cv2.flip(frame, 1)
        
        # Convertir a escala de grises para detección de rostros
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))
        
        for (x, y, w, h) in faces:
            # Extraer el rostro con margen
            margin = 20
            y1 = max(0, y - margin)
            y2 = min(frame.shape[0], y + h + margin)
            x1 = max(0, x - margin)
            x2 = min(frame.shape[1], x + w + margin)
            
            face_roi = frame[y1:y2, x1:x2]
            
            # Preprocesar la imagen
            face_tensor = transform(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
            face_tensor = face_tensor.unsqueeze(0).to(device)
            
            # Hacer predicción
            with torch.no_grad():
                outputs = model(face_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
                emotion_idx = predicted.item()
                emotion = EMOTIONS[emotion_idx]
                conf_percent = confidence.item() * 100
            
            # Color según la emoción
            color = emotion_colors.get(emotion, (0, 255, 0))
            
            # Dibujar rectángulo y etiqueta
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Fondo para texto
            label = f"{emotion}: {conf_percent:.1f}%"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x, y-30), (x + label_size[0], y), color, -1)
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Mostrar barras de probabilidad para todas las emociones
            bar_width = 150
            bar_height = 15
            start_y = y + h + 10
            
            for i, (emo, prob) in enumerate(zip(EMOTIONS, probs[0])):
                bar_length = int(bar_width * prob.item())
                bar_color = emotion_colors.get(emo, (128, 128, 128))
                
                cv2.rectangle(frame, (x, start_y + i*20), 
                            (x + bar_length, start_y + i*20 + bar_height), 
                            bar_color, -1)
                cv2.putText(frame, f"{emo[:3]}", (x + bar_width + 5, start_y + i*20 + 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Mostrar frame
        cv2.imshow('AffectNet Emotion Recognition', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('emotion_screenshot.png', frame)
            print("Screenshot guardado como 'emotion_screenshot.png'")
    
    cap.release()
    cv2.destroyAllWindows()

# Función principal
def main():
    # Rutas al dataset
    data_root = 'C:\\Users\\luuis\\OneDrive\\Desktop\\IA\\ABB\\Módulo 4\\affectnet-yolo' 
    
    # Obtener transformaciones
    train_transform, val_transform = get_transforms()
    
    # Crear datasets
    print("Cargando datasets...")
    train_dataset = AffectNetDataset(data_root, transform=train_transform, subset='train')
    val_dataset = AffectNetDataset(data_root, transform=val_transform, subset='val')
    
    # Crear dataloaders con weighted sampling para manejar desbalance
    train_sampler = get_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                            sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"\nTotal de imágenes:")
    print(f"Entrenamiento: {len(train_dataset)}")
    print(f"Validación: {len(val_dataset)}")
    
    # Crear modelo
    model = AffectNetCNN(num_classes=NUM_CLASSES, model_name='efficientnet').to(device)
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}")
    
    # Definir función de pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Scheduler de learning rate
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Entrenar modelo
    print("\nIniciando entrenamiento...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS
    )
    
    # Visualizar resultados
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # Probar con webcam
    #print("\nIniciando prueba con webcam...")
    #test_webcam()

# Función para visualizar resultados de entrenamiento
def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfica de pérdida
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfica de accuracy
    ax2.plot(train_accs, label='Train Acc', linewidth=2)
    ax2.plot(val_accs, label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('affectnet_training_history.png', dpi=300)
    plt.show()

# Función para evaluar el modelo final
def evaluate_model(model_path='best_affectnet_model.pth'):
    # Cargar modelo
    model = AffectNetCNN(num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Cargar dataset de test
    _, val_transform = get_transforms()
    test_dataset = AffectNetDataset('../affectnet-yolo', transform=val_transform, subset='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Matriz de confusión
    cm = confusion_matrix(all_labels, all_predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title('Confusion Matrix - AffectNet')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('affectnet_confusion_matrix.png', dpi=300)
    plt.show()
    
    # Reporte de clasificación
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=EMOTIONS))

if __name__ == "__main__":
    main()
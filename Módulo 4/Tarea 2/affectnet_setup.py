import os
import shutil
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# Configuración de AffectNet
EMOTIONS = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']

def verify_dataset_structure(root_dir):
    """
    Verifica la estructura del dataset AffectNet en formato YOLO
    """
    print("Verificando estructura del dataset...")
    print(f"Directorio raíz: {root_dir}")
    
    required_structure = {
        'train': ['images', 'labels'],
        'val': ['images', 'labels'],
        'test': ['images', 'labels']  # Opcional
    }
    
    dataset_stats = {
        'train': {'images': 0, 'labels': 0, 'class_dist': {}},
        'val': {'images': 0, 'labels': 0, 'class_dist': {}},
        'test': {'images': 0, 'labels': 0, 'class_dist': {}}
    }
    
    for subset, folders in required_structure.items():
        subset_path = os.path.join(root_dir, subset)
        
        if not os.path.exists(subset_path):
            print(f"⚠️  Advertencia: No se encontró el directorio '{subset}'")
            continue
        
        print(f"\n📁 Verificando {subset}:")
        
        for folder in folders:
            folder_path = os.path.join(subset_path, folder)
            
            if not os.path.exists(folder_path):
                print(f"  ❌ Falta directorio: {folder}")
                continue
            
            files = os.listdir(folder_path)
            
            if folder == 'images':
                image_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]
                dataset_stats[subset]['images'] = len(image_files)
                print(f"  ✓ {folder}: {len(image_files)} archivos")
            
            elif folder == 'labels':
                label_files = [f for f in files if f.endswith('.txt')]
                dataset_stats[subset]['labels'] = len(label_files)
                print(f"  ✓ {folder}: {len(label_files)} archivos")
                
                # Analizar distribución de clases
                class_counts = {i: 0 for i in range(len(EMOTIONS))}
                
                for label_file in tqdm(label_files[:1000], desc=f"  Analizando etiquetas {subset}"):
                    label_path = os.path.join(folder_path, label_file)
                    try:
                        with open(label_path, 'r') as f:
                            line = f.readline().strip()
                            if line:
                                class_id = int(line.split()[0])
                                if 0 <= class_id < len(EMOTIONS):
                                    class_counts[class_id] += 1
                    except:
                        pass
                
                dataset_stats[subset]['class_dist'] = class_counts
    
    return dataset_stats

def visualize_dataset_stats(dataset_stats):
    """
    Visualiza estadísticas del dataset
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Estadísticas del Dataset AffectNet', fontsize=16)
    
    # Gráfico de barras para cantidad de imágenes por subset
    ax1 = axes[0, 0]
    subsets = list(dataset_stats.keys())
    image_counts = [dataset_stats[s]['images'] for s in subsets]
    bars1 = ax1.bar(subsets, image_counts, color=['blue', 'green', 'red'])
    ax1.set_title('Número de Imágenes por Subset')
    ax1.set_ylabel('Cantidad')
    
    # Añadir etiquetas a las barras
    for bar, count in zip(bars1, image_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}', ha='center', va='bottom')
    
    # Distribución de clases en train
    ax2 = axes[0, 1]
    train_dist = dataset_stats['train']['class_dist']
    if train_dist:
        emotions = [EMOTIONS[i] for i in range(len(EMOTIONS))]
        counts = [train_dist.get(i, 0) for i in range(len(EMOTIONS))]
        bars2 = ax2.bar(emotions, counts, color='skyblue')
        ax2.set_title('Distribución de Emociones (Train)')
        ax2.set_ylabel('Cantidad')
        ax2.set_xticklabels(emotions, rotation=45, ha='right')
        
        # Porcentajes
        total = sum(counts)
        for bar, count in zip(bars2, counts):
            if total > 0:
                percentage = (count / total) * 100
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Comparación train vs val
    ax3 = axes[1, 0]
    if dataset_stats['train']['images'] > 0 and dataset_stats['val']['images'] > 0:
        sizes = [dataset_stats['train']['images'], dataset_stats['val']['images']]
        labels = ['Train', 'Validation']
        ax3.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Proporción Train vs Validation')
    
    # Tabla de resumen
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = []
    for subset in ['train', 'val', 'test']:
        if dataset_stats[subset]['images'] > 0:
            table_data.append([
                subset.capitalize(),
                f"{dataset_stats[subset]['images']:,}",
                f"{dataset_stats[subset]['labels']:,}",
                'Sí' if dataset_stats[subset]['images'] == dataset_stats[subset]['labels'] else 'No'
            ])
    
    table = ax4.table(cellText=table_data,
                      colLabels=['Subset', 'Imágenes', 'Etiquetas', 'Completo'],
                      cellLoc='center',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax4.set_title('Resumen del Dataset')
    
    plt.tight_layout()
    plt.savefig('affectnet_dataset_stats.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_sample_images(root_dir, num_samples=16):
    """
    Visualiza imágenes de muestra del dataset con sus etiquetas
    """
    train_images_dir = os.path.join(root_dir, 'train', 'images')
    train_labels_dir = os.path.join(root_dir, 'train', 'labels')
    
    if not os.path.exists(train_images_dir):
        print("No se encontró el directorio de imágenes de entrenamiento")
        return
    
    # Obtener lista de imágenes
    image_files = [f for f in os.listdir(train_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) == 0:
        print("No se encontraron imágenes")
        return
    
    # Seleccionar muestras aleatorias
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    # Crear grid de visualización
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle('Muestras del Dataset AffectNet', fontsize=16)
    
    if grid_size == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (ax, img_file) in enumerate(zip(axes, sample_files)):
        # Cargar imagen
        img_path = os.path.join(train_images_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Cargar etiqueta
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(train_labels_dir, label_file)
        
        emotion = "Unknown"
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    line = f.readline().strip()
                    if line:
                        class_id = int(line.split()[0])
                        if 0 <= class_id < len(EMOTIONS):
                            emotion = EMOTIONS[class_id]
            except:
                pass
        
        # Mostrar imagen
        ax.imshow(img)
        ax.set_title(f'{emotion}', fontsize=10)
        ax.axis('off')
    
    # Ocultar ejes sobrantes
    for idx in range(len(sample_files), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('affectnet_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

def check_image_quality(root_dir, subset='train', num_samples=100):
    """
    Verifica la calidad de las imágenes (tamaño, formato, etc.)
    """
    images_dir = os.path.join(root_dir, subset, 'images')
    
    if not os.path.exists(images_dir):
        print(f"No se encontró el directorio: {images_dir}")
        return
    
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    sizes = []
    formats = []
    corrupted = 0
    
    print(f"\nAnalizando calidad de {len(sample_files)} imágenes...")
    
    for img_file in tqdm(sample_files, desc="Verificando imágenes"):
        img_path = os.path.join(images_dir, img_file)
        
        try:
            # Abrir con PIL para verificar
            img = Image.open(img_path)
            sizes.append(img.size)
            formats.append(img.format)
            img.verify()  # Verificar integridad
        except:
            corrupted += 1
    
    # Estadísticas
    print(f"\n📊 Estadísticas de calidad:")
    print(f"  - Imágenes corruptas: {corrupted}/{len(sample_files)}")
    
    if sizes:
        sizes_array = np.array(sizes)
        print(f"  - Tamaño promedio: {np.mean(sizes_array[:, 0]):.0f}x{np.mean(sizes_array[:, 1]):.0f}")
        print(f"  - Tamaño mínimo: {np.min(sizes_array[:, 0])}x{np.min(sizes_array[:, 1])}")
        print(f"  - Tamaño máximo: {np.max(sizes_array[:, 0])}x{np.max(sizes_array[:, 1])}")
    
    if formats:
        unique_formats = list(set(formats))
        print(f"  - Formatos encontrados: {', '.join(unique_formats)}")

def prepare_dataset_from_kaggle(kaggle_dir, output_dir):
    """
    Prepara el dataset descargado de Kaggle en la estructura esperada
    """
    print("Preparando dataset desde Kaggle...")
    
    # Verificar estructura de Kaggle
    possible_structures = [
        # Estructura 1: train/val/test directamente
        {'train': 'train', 'val': 'val', 'test': 'test'},
        # Estructura 2: con subdirectorios
        {'train': 'affectnet/train', 'val': 'affectnet/val', 'test': 'affectnet/test'},
        # Estructura 3: todo en un directorio
        {'all': 'images'}
    ]
    
    found_structure = None
    for structure in possible_structures:
        if all(os.path.exists(os.path.join(kaggle_dir, path)) for path in structure.values()):
            found_structure = structure
            break
    
    if not found_structure:
        print("❌ No se pudo identificar la estructura del dataset de Kaggle")
        print("Por favor, verifica la estructura de directorios")
        return
    
    # Crear directorios de salida
    for subset in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, subset, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, subset, 'labels'), exist_ok=True)
    
    # Copiar archivos según la estructura encontrada
    if 'all' in found_structure:
        print("⚠️  Dataset no dividido encontrado. Creando división 80-20...")
        # Implementar división manual
        # ... (código de división)
    else:
        # Copiar archivos manteniendo la estructura
        for subset, src_path in found_structure.items():
            src_full_path = os.path.join(kaggle_dir, src_path)
            dst_path = os.path.join(output_dir, subset)
            
            if os.path.exists(src_full_path):
                print(f"Copiando {subset}...")
                # Copiar imágenes y etiquetas
                # ... (código de copia)

def create_emotion_distribution_plot(root_dir):
    """
    Crea un gráfico detallado de la distribución de emociones
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(EMOTIONS)))
    
    # Recopilar datos de todos los subsets
    all_stats = {}
    for subset in ['train', 'val', 'test']:
        labels_dir = os.path.join(root_dir, subset, 'labels')
        if not os.path.exists(labels_dir):
            continue
        
        class_counts = {i: 0 for i in range(len(EMOTIONS))}
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        
        for label_file in tqdm(label_files, desc=f"Analizando {subset}"):
            label_path = os.path.join(labels_dir, label_file)
            try:
                with open(label_path, 'r') as f:
                    line = f.readline().strip()
                    if line:
                        class_id = int(line.split()[0])
                        if 0 <= class_id < len(EMOTIONS):
                            class_counts[class_id] += 1
            except:
                pass
        
        all_stats[subset] = class_counts
    
    # Gráfico 1: Distribución por subset
    x = np.arange(len(EMOTIONS))
    width = 0.25
    
    for i, (subset, counts) in enumerate(all_stats.items()):
        values = [counts[j] for j in range(len(EMOTIONS))]
        ax1.bar(x + i*width, values, width, label=subset.capitalize(), alpha=0.8)
    
    ax1.set_xlabel('Emociones')
    ax1.set_ylabel('Cantidad de imágenes')
    ax1.set_title('Distribución de Emociones por Subset')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(EMOTIONS, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Distribución total (pie chart)
    if 'train' in all_stats:
        total_counts = [all_stats['train'][i] for i in range(len(EMOTIONS))]
        ax2.pie(total_counts, labels=EMOTIONS, autopct='%1.1f%%', colors=colors)
        ax2.set_title('Distribución Total de Emociones (Train)')
    
    plt.tight_layout()
    plt.savefig('affectnet_emotion_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Función principal para verificar y preparar el dataset
    """
    # Configurar rutas
    kaggle_dataset_path = input("Ingresa la ruta al dataset de Kaggle descargado: ").strip()
    if not kaggle_dataset_path:
        kaggle_dataset_path = "../affectnet-yolo"
    
    print(f"\n🔍 Verificando dataset en: {kaggle_dataset_path}")
    
    # Verificar estructura
    stats = verify_dataset_structure(kaggle_dataset_path)
    
    # Visualizar estadísticas
    print("\n📊 Generando visualizaciones...")
    visualize_dataset_stats(stats)
    
    # Mostrar muestras
    print("\n🖼️ Mostrando imágenes de muestra...")
    visualize_sample_images(kaggle_dataset_path)
    
    # Verificar calidad
    print("\n🔍 Verificando calidad de imágenes...")
    check_image_quality(kaggle_dataset_path)
    
    # Crear gráfico de distribución
    print("\n📈 Creando gráfico de distribución de emociones...")
    create_emotion_distribution_plot(kaggle_dataset_path)
    
    print("\n✅ Verificación completada!")
    print("\nPara entrenar el modelo, ejecuta:")
    print("python emotion_classifier.py")
    
    # Crear archivo de configuración
    config_content = f"""# Configuración del Dataset AffectNet
DATASET_PATH = '{kaggle_dataset_path}'
EMOTIONS = {EMOTIONS}
NUM_CLASSES = {len(EMOTIONS)}

# Estadísticas del dataset
TRAIN_SAMPLES = {stats['train']['images']}
VAL_SAMPLES = {stats['val']['images']}
TEST_SAMPLES = {stats['test']['images']}
"""
    
    with open('affectnet_config.py', 'w') as f:
        f.write(config_content)
    
    print("\n📝 Archivo de configuración creado: affectnet_config.py")

if __name__ == "__main__":
    main()
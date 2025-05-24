import os
import cv2
import re
import albumentations as A

# Rutas
INPUT_DIR = "../YOLO_format/train/images"
LABEL_DIR = "../YOLO_format/train/labels"
OUTPUT_DIR = "../Aumentada/train_augmented/images"
OUT_LABEL_DIR = "../Aumentada/train_augmented/labels"

# Número de fotos aumentadas y de imágenes a aumentar
NUM_AUGS = 10
MAX_IMAGES = 1 # Para probar ---40552 max

# Crear carpetas de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUT_LABEL_DIR, exist_ok=True)

# Transformaciones visuales
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.3),
    A.Rotate(limit=20, p=0.4),
    A.GaussNoise(p=0.2),
    A.RandomScale(scale_limit=0.2, p=0.3),
    A.Blur(blur_limit=3, p=0.2),
    A.CLAHE(p=0.2),
    A.ColorJitter(p=0.3),
    A.Resize(416, 416),
])

# Extraer número desde nombre
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

# Leer imagen y clase
def load_image_and_label(img_path, txt_path):
    img = cv2.imread(img_path)
    label = None
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            line = f.readline().strip()
            if line:
                label = int(line.split()[0])  # Solo clase
    return img, label

# Aumentar imagen y guardar con etiqueta fija
def augment_and_save(img_path, txt_path, base):
    img, label = load_image_and_label(img_path, txt_path)
    if label is None:
        return

    for i in range(NUM_AUGS):
        augmented = transform(image=img)
        aug_img = augmented['image']

        # Guardar imagen
        out_img_path = os.path.join(OUTPUT_DIR, f'{base}_aug{i}.png')
        cv2.imwrite(out_img_path, aug_img)

        # Guardar etiqueta
        out_txt_path = os.path.join(OUT_LABEL_DIR, f'{base}_aug{i}.txt')
        with open(out_txt_path, 'w') as f:
            f.write(f"{label} 0.499 0.499 0.999 0.999\n")

# Obtener imágenes ordenadas por número
image_files = sorted(
    [f for f in os.listdir(INPUT_DIR) if f.endswith('.png')],
    key=extract_number
)

# Procesar las primeras N imágenes
processed = 0
for file in image_files:
    if processed >= MAX_IMAGES:
        break
    img_path = os.path.join(INPUT_DIR, file)
    txt_path = os.path.join(LABEL_DIR, file.replace('.png', '.txt'))
    base_name = os.path.splitext(file)[0]
    augment_and_save(img_path, txt_path, base_name)
    processed += 1

import torch
from emotion_classifier import test_webcam, AffectNetCNN

# Cargar y probar el modelo
print("Iniciando cámara...")
test_webcam('best_affectnet_model.pth')
import cv2
from ultralytics import YOLO

# Charger le modèle YOLO
model = YOLO("yolov8n.pt")  # Tu peux remplacer par 'yolov8s.pt', 'yolov8m.pt', etc.

# Initialiser la caméra (utilise 0 pour la webcam par défaut)
cap = cv2.VideoCapture(2)

# Vérifie que la caméra est disponible
if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la caméra.")
    exit()

while True:
    # Capture une frame depuis la caméra
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire la frame.")
        break

    # Effectuer une détection sur la frame
    results = model(frame)

    # Dessiner les résultats sur la frame
    annotated_frame = results[0].plot()

    # Afficher la frame annotée
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Quitte si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()

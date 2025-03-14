import cv2
import os
import yt_dlp
from ultralytics import YOLO
from datetime import datetime
import param  # Importation des paramètres

# Charger le modèle YOLO depuis param.py
model = YOLO(param.YOLO_MODEL)

# Dossier de sortie
os.makedirs(param.OUTPUT_DIR, exist_ok=True)

# Dictionnaire pour suivre les IDs des objets détectés
object_counter = {}

# Fonction pour récupérer l'URL du stream YouTube
def get_youtube_stream_url(youtube_url):
    ydl_opts = {"format": "best[ext=mp4]"}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info["url"]
    except Exception as e:
        print(f"Erreur lors de l'extraction du flux YouTube : {e}")
        return None

# Sélection de la source vidéo
print("\nChoisissez une source vidéo:")
print("[1] Webcam")
print("[2] Vidéo YouTube")
print("[3] Vidéo locale")
print("[4] Image unique ou dossier d'images")

choice = input("Entrez le numéro correspondant : ")

if choice == "1":
    source = param.DEFAULT_SOURCE
elif choice == "2":
    youtube_url = input("Entrez l'URL YouTube : ")
    source = get_youtube_stream_url(youtube_url)
    if not source:
        print("Erreur : Impossible de récupérer le flux YouTube.")
        exit()
elif choice == "3":
    source = input("Entrez le chemin de la vidéo locale : ")
    if not os.path.isfile(source):
        print("Erreur : Fichier vidéo introuvable.")
        exit()
elif choice == "4":
    source = input("Entrez le chemin du fichier image ou du dossier : ")
    if not os.path.exists(source):
        print("Erreur : Fichier ou dossier introuvable.")
        exit()
else:
    print("Choix invalide, utilisation de la webcam par défaut.")
    source = param.DEFAULT_SOURCE

print(f"Source sélectionnée : {source}")

# Vérifier si c'est une image ou une vidéo
if isinstance(source, str) and os.path.isfile(source) and source.lower().endswith(('.png', '.jpg', '.jpeg')):
    # Détection sur une image unique
    output_image_path = os.path.join(param.OUTPUT_DIR, f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    results = model.predict(source=source, save=True, show=param.DISPLAY_REAL_TIME, conf=param.CONFIDENCE_THRESHOLD, iou=param.IOU_THRESHOLD)
    
    # Sauvegarde de l'image traitée
    results[0].save(output_image_path)
    print(f"Image traitée sauvegardée dans : {output_image_path}")

elif isinstance(source, str) and os.path.isdir(source):
    # Détection sur un dossier d’images
    model.predict(source=source, save=True, show=param.DISPLAY_REAL_TIME, conf=param.CONFIDENCE_THRESHOLD, iou=param.IOU_THRESHOLD)
    print(f"Images traitées sauvegardées dans : {param.OUTPUT_DIR}/")

else:
    # Traitement vidéo (Webcam ou vidéo)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo.")
        exit()

    writer = None
    output_video_path = os.path.join(param.OUTPUT_DIR, "output_video.mp4")

    if param.SAVE_PROCESSED_VIDEO:
        # Configuration de l'enregistrement vidéo
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Exécuter YOLO sur chaque frame avec le tracking
        results = model.track(frame, persist=True, conf=param.CONFIDENCE_THRESHOLD, iou=param.IOU_THRESHOLD, tracker=param.TRACKER, device="cuda" if param.USE_CUDA else "cpu", half=True)
        
        # Obtenir les annotations
        for det in results[0].boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])  # Coordonnées bbox
            cls = int(det.cls[0])  # ID de la classe
            cls_name = model.names[cls]  # Nom de la classe détectée

            # Renommer selon OBJECT_LABELS de param.py
            cls_name = param.OBJECT_LABELS.get(cls_name, cls_name)

            # Assigner un numéro unique
            if cls_name not in object_counter:
                object_counter[cls_name] = 1
            else:
                object_counter[cls_name] += 1

            label = f"{cls_name} {object_counter[cls_name]}"  # Exemple : "Personne 1"

            # Récupérer la couleur définie dans param.py ou utiliser la couleur par défaut
            color = param.BOX_COLORS.get(cls_name.lower(), param.DEFAULT_BOX_COLOR)

            # Dessiner la boîte et afficher le texte
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, param.TEXT_FONT, param.TEXT_COLOR, param.TEXT_THICKNESS)

            # Sauvegarde des images des objets détectés
            if param.SAVE_OBJECT_IMAGES:
                object_img = frame[y1:y2, x1:x2]
                if object_img.size > 0:
                    object_img_path = os.path.join(param.OUTPUT_DIR, f"{cls_name}_{object_counter[cls_name]}.jpg")
                    cv2.imwrite(object_img_path, object_img)

        # Enregistrement de la vidéo
        if param.SAVE_PROCESSED_VIDEO:
            writer.write(frame)

        # Affichage en temps réel si activé
        if param.DISPLAY_REAL_TIME:
            cv2.imshow("YOLOv8 Tracking", frame)

        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if param.SAVE_PROCESSED_VIDEO:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Vidéo sauvegardée dans : {output_video_path}")

print("Détection terminée !")

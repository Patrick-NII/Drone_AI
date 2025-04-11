import cv2
import os
import yt_dlp
from ultralytics import YOLO
from datetime import datetime
import param 


# Fonction pour regrouper les classes selon les LABEL_GROUPS
def get_simplified_label(cls_name):
    for group, class_list in param.LABEL_GROUPS.items():
        if cls_name in class_list:
            return group
    return cls_name  # fallback

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

# Charger le modèle YOLO
model = YOLO(param.YOLO_MODEL)

# Créer le dossier de sortie si nécessaire
os.makedirs(param.OUTPUT_DIR, exist_ok=True)

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

# Cas 1 : Image unique
if isinstance(source, str) and os.path.isfile(source) and source.lower().endswith(('.png', '.jpg', '.jpeg')):
    output_image_path = os.path.join(param.OUTPUT_DIR, f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    results = model.predict(source=source, save=True, show=param.DISPLAY_REAL_TIME,
                            conf=param.CONFIDENCE_THRESHOLD, iou=param.IOU_THRESHOLD)
    results[0].save(output_image_path)
    print(f"Image traitée sauvegardée dans : {output_image_path}")

# Cas 2 : Dossier d’images
elif isinstance(source, str) and os.path.isdir(source):
    model.predict(source=source, save=True, show=param.DISPLAY_REAL_TIME,
                  conf=param.CONFIDENCE_THRESHOLD, iou=param.IOU_THRESHOLD)
    print(f"Images traitées sauvegardées dans : {param.OUTPUT_DIR}/")

# Cas 3 : Vidéo ou webcam
else:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo.")
        exit()

    writer = None
    output_video_path = os.path.join(param.OUTPUT_DIR, "output_video.mp4")

    if param.SAVE_PROCESSED_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        
        # ✅ Récupération du FPS réel ou valeur par défaut
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] FPS original détecté : {fps}")
        if fps is None or fps <= 1.0:
            fps = 25.0  # Valeur par défaut si non détectée
            print(f"[INFO] FPS par défaut appliqué : {fps}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Tracking YOLO avec ByteTrack
        results = model.track(frame, persist=True, conf=param.CONFIDENCE_THRESHOLD,
                              iou=param.IOU_THRESHOLD, tracker=param.TRACKER,
                              device="cuda" if param.USE_CUDA else "cpu", half=True)

        # Boucle sur les objets détectés
        for det in results[0].boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            cls = int(det.cls[0])
            cls_name = model.names[cls]

            label = get_simplified_label(cls_name)
            color = param.BOX_COLORS.get(label, param.DEFAULT_BOX_COLOR)

            # Dessiner la bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Afficher le texte
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        param.TEXT_FONT, param.TEXT_COLOR, param.TEXT_THICKNESS)

            # Sauvegarder les images si activé
            if param.SAVE_OBJECT_IMAGES:
                object_img = frame[y1:y2, x1:x2]
                if object_img.size > 0:
                    object_img_path = os.path.join(param.OUTPUT_DIR, f"{label}_{datetime.now().strftime('%H%M%S%f')}.jpg")
                    cv2.imwrite(object_img_path, object_img)

        if param.SAVE_PROCESSED_VIDEO:
            writer.write(frame)

        if param.DISPLAY_REAL_TIME:
            cv2.imshow("YOLOv8 Tracking - Secourisme", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if param.SAVE_PROCESSED_VIDEO:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Vidéo sauvegardée dans : {output_video_path}")

print("Détection terminée !")
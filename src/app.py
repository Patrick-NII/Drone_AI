import cv2
import os
import yt_dlp
from ultralytics import YOLO
from datetime import datetime

# Charger le modèle YOLO
model = YOLO("yolo11x.pt")  # Charge le modèle
# model = YOLO("yolo11x-seg.pt")  # Modèle segmentation
# model = YOLO("yolo11n-pose.pt")  # Modèle pose estimation

# Dossier de sortie
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# Sélection de la source
print("\nChoisissez une source vidéo:")
print("[1] Webcam")
print("[2] Vidéo YouTube")
print("[3] Vidéo locale")
print("[4] Image unique ou dossier d'images")

choice = input("Entrez le numéro correspondant : ")

if choice == "1":
    source = 0  # Webcam
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
    source = 0

print(f"Source sélectionnée : {source}")

# Vérifier si c'est une image ou une vidéo
if isinstance(source, str) and os.path.isfile(source) and source.lower().endswith(('.png', '.jpg', '.jpeg')):
    # Détection sur une image unique
    output_image_path = os.path.join(OUTPUT_DIR, f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    results = model.predict(source=source, save=True, show=True, conf=0.7, iou=0.65)
    
    # Sauvegarde de l'image traitée
    results[0].save(output_image_path)
    print(f"Image traitée sauvegardée dans : {output_image_path}")

elif isinstance(source, str) and os.path.isdir(source):
    # Détection sur un dossier d’images
    model.predict(source=source, save=True, show=True, conf=0.7, iou=0.65)
    print(f"Images traitées sauvegardées dans : {OUTPUT_DIR}/")

else:
    # Traitement vidéo (Webcam ou vidéo)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo.")
        exit()

    writer = None
    output_video_path = os.path.join(OUTPUT_DIR, "output_video.mp4")

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

        # Exécuter YOLO sur chaque frame avec les améliorations
        results = model.track(frame, persist=True, conf=0.7, iou=0.65, tracker="bytetrack.yaml", device="cuda", half=True)
        annotated_frame = results[0].plot()

        # Enregistrement de la vidéo
        writer.write(annotated_frame)

        # Affichage en temps réel
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Vidéo sauvegardée dans : {output_video_path}")

print("Détection terminée !")

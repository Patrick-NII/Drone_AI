from ultralytics import YOLO  # YOLOv8
import cv2  # OpenCV
import yt_dlp  # Pour télécharger les vidéos YouTube
import os  # Gestion des fichiers

#  Charger le modèle YOLO (remplace par ton modèle si besoin)
model = YOLO("yolov8n.pt")

#  Fonction pour récupérer l'URL du stream YouTube
def get_youtube_stream_url(youtube_url):
    ydl_opts = {
        "format": "best[ext=mp4]",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]

# 🎛️ Sélection de la source
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
    source = get_youtube_stream_url(youtube_url)  # Récupérer le flux
elif choice == "3":
    source = input("Entrez le chemin de la vidéo locale : ")
elif choice == "4":
    source = input("Entrez le chemin du fichier image ou du dossier : ")
else:
    print(" Choix invalide, utilisation de la webcam par défaut.")
    source = 0

print(f"Source sélectionnée : {source}")

# 📌 Vérifier si c'est une image ou une vidéo
if isinstance(source, str) and os.path.isfile(source) and source.lower().endswith(('.png', '.jpg', '.jpeg')):
    # 📷 Détection sur une image unique
    results = model.predict(source=source, save=True, show=True)

elif isinstance(source, str) and os.path.isdir(source):
    # 📂 Détection sur un dossier d’images
    results = model.predict(source=source, save=True, show=True)

else:
    # 🎥 Traitement vidéo (Webcam ou vidéo)
    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Exécuter YOLO sur chaque frame
        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()

        # 📌 Affichage en temps réel
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

print(" Détection terminée !")

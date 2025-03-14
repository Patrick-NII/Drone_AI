# param.py : Paramètres de configuration pour YOLO

# ==========================
#  PARAMÈTRES YOLO
# ==========================

# Modèle YOLO à utiliser (tu peux changer selon tes besoins)
YOLO_MODEL = "yolo11x.pt"  

# Seuil de confiance pour garder une détection (0.5 = équilibré, 0.3 = plus permissif)
CONFIDENCE_THRESHOLD = 0.6  

# Seuil d'Intersection over Union (IoU) (0.6 = bon équilibre précision/rappel)
IOU_THRESHOLD = 0.6  

# Tracker utilisé pour le suivi des objets
TRACKER = "bytetrack.yaml"

# Activer CUDA pour accélérer le traitement avec le GPU
USE_CUDA = True  


# ==========================
#  SOURCES & SORTIES
# ==========================

# Source par défaut (0 = Webcam)
DEFAULT_SOURCE = 0  

# Dossier pour sauvegarder les vidéos et images détectées
OUTPUT_DIR = "output"  


# ==========================
#  PARAMÈTRES D'AFFICHAGE
# ==========================

# Définition des couleurs pour chaque type d'objet (BGR format pour OpenCV)
BOX_COLORS = {
    "person": (0, 255, 0),         # Vert
    "car": (255, 0, 0),            # Bleu
    "truck": (0, 0, 255),          # Rouge
    "motorcycle": (255, 165, 0),   # Orange
    "bicycle": (128, 0, 128),      # Violet
    "dog": (0, 255, 255),          # Cyan
    "cat": (255, 192, 203),        # Rose
    "bird": (75, 0, 130),          # Indigo
    "boat": (255, 255, 0),         # Jaune
    "traffic light": (0, 128, 128) # Bleu foncé
}

# Couleur par défaut si l'objet n'est pas dans la liste
DEFAULT_BOX_COLOR = (200, 200, 200)  # Gris clair

# Paramètres du texte
TEXT_COLOR = (0, 0, 255)  # Rouge
TEXT_FONT = 0.6  # Taille du texte pour les labels
TEXT_THICKNESS = 2  # Épaisseur du texte


# ==========================
#  DICTIONNAIRE DES OBJETS
# ==========================

OBJECT_LABELS = {
    "person": "Personne",
    "car": "Voiture",
    "truck": "Camion",
    "motorcycle": "Moto",
    "bicycle": "Vélo",
    "bus": "Bus",
    "train": "Train",
    "boat": "Bateau",
    "airplane": "Avion",
    "traffic light": "Feu de signalisation",
    "stop sign": "Panneau Stop",
    "parking meter": "Parcmètre",
    "bench": "Banc",
    "dog": "Chien",
    "cat": "Chat",
    "horse": "Cheval",
    "bird": "Oiseau",
    "cow": "Vache",
    "sheep": "Mouton",
    "elephant": "Éléphant",
    "zebra": "Zèbre",
    "giraffe": "Girafe",
    "backpack": "Sac à dos",
    "umbrella": "Parapluie",
    "handbag": "Sac à main",
    "tie": "Cravate",
    "suitcase": "Valise",
    "frisbee": "Frisbee",
    "skis": "Skis",
    "snowboard": "Snowboard",
    "sports ball": "Ballon de sport",
    "kite": "Cerf-volant",
    "baseball bat": "Batte de baseball",
    "baseball glove": "Gant de baseball",
    "skateboard": "Skateboard",
    "surfboard": "Surf",
    "tennis racket": "Raquette de tennis",
    "bottle": "Bouteille",
    "wine glass": "Verre à vin",
    "cup": "Tasse",
    "fork": "Fourchette",
    "knife": "Couteau",
    "spoon": "Cuillère",
    "bowl": "Bol",
    "banana": "Banane",
    "apple": "Pomme",
    "sandwich": "Sandwich",
    "orange": "Orange",
    "broccoli": "Brocoli",
    "carrot": "Carotte",
    "hot dog": "Hot-dog",
    "pizza": "Pizza",
    "donut": "Beignet",
    "cake": "Gâteau",
    "chair": "Chaise",
    "couch": "Canapé",
    "potted plant": "Plante en pot",
    "bed": "Lit",
    "dining table": "Table à manger",
    "toilet": "Toilettes",
    "tv": "Télévision",
    "laptop": "Ordinateur portable",
    "mouse": "Souris",
    "remote": "Télécommande",
    "keyboard": "Clavier",
    "cell phone": "Téléphone portable",
    "microwave": "Micro-ondes",
    "oven": "Four",
    "toaster": "Grille-pain",
    "sink": "Évier",
    "refrigerator": "Réfrigérateur",
    "book": "Livre",
    "clock": "Horloge",
    "vase": "Vase",
    "scissors": "Ciseaux",
    "teddy bear": "Ours en peluche",
    "hair drier": "Sèche-cheveux",
    "toothbrush": "Brosse à dents"
}

# ==========================
#  OPTIONS SUPPLÉMENTAIRES
# ==========================

# Activer l'affichage en temps réel
DISPLAY_REAL_TIME = True  

# Enregistrer la vidéo traitée
SAVE_PROCESSED_VIDEO = True  

# Enregistrer les images des objets détectés
SAVE_OBJECT_IMAGES = False  

# Ajuster la taille des images de sortie (False pour garder l'original)
RESIZE_OUTPUT = False  
OUTPUT_IMAGE_SIZE = (800, 600)  # (Largeur, Hauteur)

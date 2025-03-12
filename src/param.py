import cv2
import numpy as np
import mediapipe as mp

# Initialisation de MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialisation de la caméra
cap = cv2.VideoCapture(2)

# Création d'un canevas pour dessiner
canvas = None
drawing_color = (0, 255, 0)  # Couleur du dessin (vert)
brush_size = 5

# Variable pour suivre l'état du dessin
drawing = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Retourne l'image pour qu'elle soit comme un miroir
    frame = cv2.flip(frame, 1)

    # Création du canevas si nécessaire
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Conversion en RGB pour MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processer l'image pour détecter les mains
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dessiner les landmarks sur l'image
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraire les coordonnées de l'index
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

            # Vérifier si l'utilisateur dessine (proche du pouce)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
            distance = np.sqrt((x - thumb_x)**2 + (y - thumb_y)**2)

            # Si l'index et le pouce sont proches, commencer à dessiner
            if distance < 40:
                drawing = True
            else:
                drawing = False

            # Dessiner si l'utilisateur est en mode "drawing"
            if drawing:
                cv2.circle(canvas, (x, y), brush_size, drawing_color, -1)

    # Combiner le canevas et la vidéo
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Afficher le résultat
    cv2.imshow("Virtual Drawing", combined)

    # Commandes clavier
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quitter
        break
    elif key == ord('c'):  # Effacer le canevas
        canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np

# Ouvrir la vidéo
cap = cv2.VideoCapture('sources/vid6.mp4')

# Initialiser le modèle de soustraction de fond
fgbg = cv2.createBackgroundSubtractorMOG2()

# Obtenir les dimensions originales de la vidéo
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Obtenir la largeur et la hauteur de l'écran
screen_width = 1920  # Remplacez par la largeur de votre écran
screen_height = 1080  # Remplacez par la hauteur de votre écran

# Créer une fenêtre sans bordures redimensionnable
cv2.namedWindow('Tennis Ball Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Tennis Ball Detection', screen_width, screen_height)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Calculer les marges à enlever (5%)
    margin_x = int(0.05 * frame_width)
    margin_y = int(0.05 * frame_height)

    # Recadrer l'image pour enlever les bords
    cropped_frame = frame[margin_y:frame_height-margin_y, margin_x:frame_width-margin_x]

    # Appliquer le modèle de soustraction de fond
    fgmask = fgbg.apply(cropped_frame)

    # Filtrer le bruit en utilisant l'ouverture et la fermeture
    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    # Détection de la balle dans le masque résultant
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Filtrer les petits contours
        if  200> cv2.contourArea(contour)>40:
           # Approximation de forme pour détecter la circularité
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * cv2.contourArea(contour) / (perimeter * perimeter)

            # Filtrer les contours circulaires
            if 0.7 <= circularity <= 1.0:
                # Dessiner un rectangle autour de la balle
                x, y, w, h = cv2.boundingRect(contour)
                x += margin_x
                y += margin_y
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Tennis Ball Detection', frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
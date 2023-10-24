import cv2
import numpy as np

def detection(cap, frame):
    # Initialiser le modèle de soustraction de fond
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # Obtenir les dimensions originales de la vidéo
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

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
                
        return frame
# Function to detect a small yellow ball in a frame
def detect_yellow_ball(cap, fraame):
    
    frame = detection(cap, fraame)
    
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the yellow color
    lower_yellow = np.array([80, 100, 80])
    upper_yellow = np.array([100, 255, 255])

    # Create a mask to isolate the yellow color
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    cv2.imshow('Mask', mask)
    # Perform morphological operations to clean up the mask (optional)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_position = None

    # If contours are found, find the largest one (assumed to be the ball)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 0.5:
            # Find the center of the detected ball
            M = cv2.moments(largest_contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            detected_position = (cX, cY)

    return detected_position

# Capture the video
cap = cv2.VideoCapture('sources/vid2.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect the small yellow ball in the frame
    detected_position = detect_yellow_ball(cap, frame)

    if detected_position is not None:
        x, y = detected_position
        cv2.circle(frame, (x, y), 20, (0, 255, 0), -1)  # Draw a green circle at the detected position

    cv2.imshow('Ball Detection', frame)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

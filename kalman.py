import numpy as np
import cv2
import sys
import time
bgr_color = 29,2,128 #129,119,195
color_threshold = 50 #color range

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

#hsv_color = cv2.cvtColor( np.uint8([[bgr_color]] ), cv2.COLOR_BGR2HSV)[0][0]
#HSV_lower = np.array([hsv_color[0] - color_threshold, hsv_color[1] - color_threshold, hsv_color[2] - color_threshold])
#HSV_upper = np.array([hsv_color[0] + color_threshold, hsv_color[1] + color_threshold, hsv_color[2] + color_threshold])
HSV_lower = np.array([80, 10, 80])
HSV_upper = np.array([100, 40, 110])

def detect_ball(fraame):
    x, y, radius = -1, -1, -1
    frame = detection(cap, fraame)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_frame, HSV_lower, HSV_upper)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)

    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = (-1, -1)

    # only proceed if at least one contour was found
    if len(contours) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(mask)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) #position of the ball

        # check that the radius is larger than some threshold
        if radius > 1: #CHANGED
            #outline ball
            cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
            #show ball center
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

    return center[0], center[1], radius


if __name__ == "__main__":
    filepath = 'video_input2.mp4'

    cap = cv2.VideoCapture(filepath)

    dist = []
    times = []
    count = 0
    kalmanFilter = []
    variance = []
    initialEstimatedVariance = 1
    deltaX = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    deltat = 524  # num frames(varies per video test case)
    noise = 1  # noise Q
    sensorCovariance = 83791.65209996712



    variance.append(initialEstimatedVariance)
    #start = time.clock()
    while(cap.isOpened()):
        count+=1
        # Capture frame-by-frame
        ret, frame = cap.read()

        #detect_ball(frame)
        [x,y,r] = detect_ball(frame)
        #t = time.clock() - start
        if (len(dist) != 0):
            x_prev = dist[len(dist) - 1]

            velocity = float(deltaX)/deltat

            dist.append(x)

            estimatedX = x_prev + velocity
            estimatedVariance = variance[len(variance)-1] + noise

            kalmanGain = float(estimatedVariance)/(estimatedVariance + sensorCovariance)

            kalmanPosition = estimatedX + (kalmanGain*(x - estimatedX))

            kalmanFilter.append(kalmanPosition)

            updatedVariance = estimatedVariance-(kalmanGain*estimatedVariance)
            variance.append(updatedVariance)

            print (kalmanFilter)
            #print(variance)

        else: #don't do any calculations on the 1st point
            x_initial = detect_ball(frame)[0]
            dist.append(x_initial)
            kalmanFilter.append(x_initial)

        # Display the resulting frame
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
        imS = cv2.resizeWindow('frame', (960, 540))  # Resize image

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
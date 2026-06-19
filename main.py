import cv2
import mediapipe as mp
import numpy as np

# ==========================
# MediaPipe Setup
# ==========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

# ==========================
# Webcam
# ==========================
cap = cv2.VideoCapture(0)

# ==========================
# Variables
# ==========================
canvas = None
xp, yp = 0, 0

drawColor = (255, 0, 255)

brushThickness = 8
eraserThickness = 35

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    h, w, c = frame.shape

    # Create canvas once
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # ==========================
    # Toolbar
    # ==========================

    cv2.rectangle(frame, (0, 0), (100, 65), (255, 0, 0), cv2.FILLED)
    cv2.rectangle(frame, (100, 0), (200, 65), (0, 255, 0), cv2.FILLED)
    cv2.rectangle(frame, (200, 0), (300, 65), (0, 0, 255), cv2.FILLED)
    cv2.rectangle(frame, (300, 0), (400, 65), (0, 0, 0), cv2.FILLED)
    cv2.rectangle(frame, (400, 0), (500, 65), (0, 255, 255), cv2.FILLED)
    cv2.rectangle(frame, (500, 0), (620, 65), (128, 128, 128), cv2.FILLED)

    cv2.putText(frame, "Blue", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, "Green", (110, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, "Red", (220, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, "Erase", (310, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, "Yellow", (405, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.putText(frame, "Clear", (525, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ==========================
    # Hand Detection
    # ==========================

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Index Finger Tip
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            cv2.circle(frame, (x, y), 15,
                       (0, 255, 255), cv2.FILLED)

            # ==========================
            # Color Selection
            # ==========================

            if y < 65:

                if 0 < x < 100:
                    drawColor = (255, 0, 0)

                elif 100 < x < 200:
                    drawColor = (0, 255, 0)

                elif 200 < x < 300:
                    drawColor = (0, 0, 255)

                elif 300 < x < 400:
                    drawColor = (0, 0, 0)

                elif 400 < x < 500:
                    drawColor = (0, 255, 255)

                elif 500 < x < 620:
                    canvas = np.zeros((h, w, 3), dtype=np.uint8)

                xp, yp = 0, 0

            # ==========================
            # Drawing
            # ==========================

            else:

                if xp == 0 and yp == 0:
                    xp, yp = x, y

                thickness = (
                    eraserThickness
                    if drawColor == (0, 0, 0)
                    else brushThickness
                )

                cv2.line(
                    canvas,
                    (xp, yp),
                    (x, y),
                    drawColor,
                    thickness
                )

                xp, yp = x, y

    # ==========================
    # Merge Canvas
    # ==========================

    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    _, imgInv = cv2.threshold(
        imgGray,
        50,
        255,
        cv2.THRESH_BINARY_INV
    )

    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    frame = cv2.bitwise_and(frame, imgInv)
    frame = cv2.bitwise_or(frame, canvas)

    # ==========================
    # Show Window
    # ==========================

    cv2.imshow("Air Canvas", frame)

    key = cv2.waitKey(1)

    # Save Drawing
    if key == ord('s'):
        cv2.imwrite("drawing.png", canvas)
        print("Drawing saved as drawing.png")

    # Increase Brush Size
    if key == ord('+') or key == ord('='):
        brushThickness += 2
        print("Brush Size:", brushThickness)

    # Decrease Brush Size
    if key == ord('-'):
        brushThickness = max(2, brushThickness - 2)
        print("Brush Size:", brushThickness)

    # Exit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
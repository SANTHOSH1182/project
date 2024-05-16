import cv2
import face_recognition
import pygame
import time
import autopy
import mediapipe
import numpy

# Load known face image and encode it
known_image = face_recognition.load_image_file("nirmal.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Initialize pygame for displaying the alert
pygame.init()

# Set up alert window
alert_font = pygame.font.SysFont(None, 30)
alert_display = pygame.display.set_mode((400, 100))
pygame.display.set_caption('Face Recognition Alert')

# Initialize variables for popup display
show_popup = False
popup_start_time = None
popup_duration = 5  # in seconds
automation_start_time = None
automation_duration = 2  # in seconds

# Initialize variable for authorization
authorized = False

# Open the webcam
cap = cv2.VideoCapture(0)

# Initialize hand tracking
init_hand = mediapipe.solutions.hands
main_hand = init_hand.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
draw = mediapipe.solutions.drawing_utils

pX, pY = 0, 0  # Previous x and y location
w_scr, h_scr = autopy.screen.size()  # Outputs the high and width of the screen (1920 x 1080)


def hand_landmarks(color_img):
    landmark_list = []  # Default values if no landmarks are tracked
    landmark_positions = main_hand.process(color_img)  # Object for processing the video input
    landmark_check = landmark_positions.multi_hand_landmarks  # Stores the out of the processing object (returns False on empty)

    if landmark_check:  # Checks if landmarks are tracked
        for hand in landmark_check:  # Landmarks for each hand
            for index, landmark in enumerate(
                    hand.landmark):  # Loops through the 21 indexes and outputs their landmark coordinates (x, y, & z)
                draw.draw_landmarks(color_img, hand,
                                    init_hand.HAND_CONNECTIONS)  # Draws each individual index on the hand with connections
                h, w, c = color_img.shape  # Height, width, and channel on the image
                centerX, centerY = int(landmark.x * w), int(
                    landmark.y * h)  # Converts the decimal coordinates relative to the image for each index
                landmark_list.append([index, centerX, centerY])  # Adding index and its coordinates to a list

    return landmark_list


def fingers(landmarks):
    finger_tips = []  # To store 4 sets of 1s or 0s
    tip_ids = [4, 8, 12, 16, 20]  # Indexes for the tips of each finger

    # Check if thumb is up
    if landmarks[tip_ids[0]][1] > landmarks[tip_ids[0] - 1][1]:
        finger_tips.append(1)
    else:
        finger_tips.append(0)

    # Check if fingers are up except the thumb
    for id in range(1, 5):
        if landmarks[tip_ids[id]][2] < landmarks[tip_ids[id] - 3][
            2]:  # Checks to see if the tip of the finger is higher than the joint
            finger_tips.append(1)
        else:
            finger_tips.append(0)

    return finger_tips


while True:
    ret, frame = cap.read()

    if not authorized:
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces([known_encoding], face_encoding)
            name = "Unknown"

            if True in matches:
                name = "Nirmal kumar"
                authorized = True
                if not show_popup:
                    show_popup_message('Authorized Person Detected!', text_color=(0, 255, 0))
                    time.sleep(1.5)
                    show_popup_message('Starting Automation in 2 seconds...', text_color=(0, 255, 150))
                    automation_start_time = time.time()

            else:
                authorized = False
                show_popup_message('Unauthorized Person Detected!', text_color=(255, 0, 0))
                time.sleep(5)

        if show_popup and time.time() - popup_start_time > popup_duration:
            show_popup = False

    else:  # User is authorized
        lm_list = hand_landmarks(frame)

        if len(lm_list) != 0:
            x1, y1 = lm_list[8][1:]
            x2, y2 = lm_list[12][1:]
            finger = fingers(lm_list)

            if finger[1] == 1 and finger[2] == 0:
                x3 = numpy.interp(x1, (75, 640 - 75), (0, w_scr))
                y3 = numpy.interp(y1, (75, 480 - 75), (0, h_scr))

                cX = pX + (x3 - pX) / 7
                cY = pY + (y3 - pY) / 7

                autopy.mouse.move(w_scr - cX, cY)
                pX, pY = cX, cY

            if finger[1] == 0 and finger[0] == 1:
                autopy.mouse.click()

    cv2.imshow("Webcam", frame)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()

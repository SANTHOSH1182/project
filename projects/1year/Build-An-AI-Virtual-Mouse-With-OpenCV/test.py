import cv2
import face_recognition
import pygame
import time

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

# Function to display the popup
def show_popup_message(message, text_color=(255, 255, 255)):
    global show_popup, popup_start_time
    show_popup = True
    popup_start_time = time.time()
    alert_text = alert_font.render(message, True, text_color)
    alert_display.fill((0, 0, 0))
    alert_display.blit(alert_text, (10, 30))
    pygame.display.update()

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Find face locations in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the current face matches the known face
        matches = face_recognition.compare_faces([known_encoding], face_encoding)
        name = "Unknown"

        if True in matches:
            name = "Nirmal kumar"
            # Display popup if the face is recognized for the first time
            if not show_popup:
                show_popup_message('Authorized Person Detected!',text_color=(0, 255, 0))
                # Wait for 3 seconds after detecting an authorized person
                time.sleep(1.5)
                # Display an alert before starting automation
                show_popup_message('Starting Automation in 2 seconds...', text_color=(0, 255, 150))
                # Record the start time for automation
                automation_start_time = time.time()
        else:
            # Display popup in red if the face is unrecognized
            show_popup_message('Unauthorized Person Detected!', text_color=(255, 0, 0))
            time.sleep(5)

        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Close popup after a certain duration
    if show_popup and time.time() - popup_start_time > popup_duration:
        show_popup = False

    # Check if it's time to start automation
    if automation_start_time and time.time() - automation_start_time > automation_duration:
        print("Starting Automation...")
        # Add your automation code here
        # ...

    # Handle events in the popup window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
pygame.quit()  # Close the pygame window

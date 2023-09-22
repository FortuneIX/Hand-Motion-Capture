import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Create a window to display the video
cv2.namedWindow('Hand Motion Detection', cv2.WINDOW_NORMAL)

# Initialize the previous frame for motion detection
ret, prev_frame = cap.read()

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define a range of skin color values in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask to extract the skin color region
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological operations to clean up the mask
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=2)

    # Bitwise-AND the mask and the original frame
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Perform motion detection using frame differencing
    diff = cv2.absdiff(prev_frame, frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around moving objects (potential hand motion)
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the processed frame
    cv2.imshow('Hand Motion Detection', frame)

    # Store the current frame for the next iteration
    prev_frame = frame

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

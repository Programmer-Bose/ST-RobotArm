import cv2

# Create a VideoCapture object to access the default camera (index 0)
cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()
else:
    print("Webcam successfully accessed. Press 'q' to quit.")

while True:
    # Read a frame from the camera
    # 'ret' is a boolean (True/False) indicating success
    # 'frame' is the image array (a NumPy array)
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Display the resulting frame in a window named 'Webcam Feed'
    cv2.imshow('Webcam Feed', frame)

    # Wait for 1 millisecond for a key press
    # If the 'q' key is pressed, break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

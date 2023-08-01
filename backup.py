import cv2
import numpy as np
import mediapipe as mp
from mediapipe import *

mp_drawing = solutions.drawing_utils
mp_pose = solutions.pose

# Get the video feed from the camera and make detections
cap = cv2.VideoCapture(0)  # We are grabbing or setting the vide capture device. The 0 represent the webcam

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:  # 0.5 are percentages(50%)
    while cap.isOpened():  # it loops through the feed
        ret, frame = cap.read()  # ret is just a return variable, frame is giving as the image from the cam

        # Detect stuff and render

        # Recolor image. Normally in CV2 in comes in format BGR.
        # In mediapipe they have to be reordered. Media pipe needs it in RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # we set some performance of the streaming,
        # by setting it to false we save a bunch of memory somehow

        # This line actually makes the detection
        results = pose.process(image)

        image.flags.writeable = True
        # Recoloring back the image. Open cv wants it in BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # print(results)
        print(results.pose_landmarks)
        print(mp_pose.POSE_CONNECTIONS)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  # landmark drawing specs(dots)
                                  mp_drawing.DrawingSpec(color=(245, 66, 280), thickness=2, circle_radius=2)
                                  # connection drawing specs (lines)
                                  )

        flipped_video = cv2.flip(image, 1)  # mirror image of the feed
        cv2.imshow("Mediapipe Feed", flipped_video)  # This gives as a popup on our screen with the visualisation.
        # We pass the Title and the image

        if cv2.waitKey(10) & 0xFF == ord("q"):  # Listener for the quit button. in this case is the letter 'q'
            break
    cap.release()  # It releases the capture device
    cv2.destroyWindow()  # and destroys the window

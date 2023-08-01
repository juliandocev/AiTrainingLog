import cv2
from mediapipe import *
import numpy as np


def calculate_angle(a, b, c):
    # takes the points and convert them tu np array
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    # calculate the radians for the particular joint
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])

    # calculation of the angle
    angle = np.abs(radians * 180.0 / np.pi)

    # convert it to an angle between 0 and 180 degree
    if angle > 180.0:
        angle = 360 - angle

    return angle


class MediaPipeTool:

    def __init__(self):
        self.mp_pose = solutions.pose
        self.mp_drawing = solutions.drawing_utils
        self.color_dots = (245, 117, 66)
        self.color_lines = (245, 66, 280)
        self.min_det_conf = 0.5  # minimum detection confidence in %. Exp: 0.5 are 50%
        self.min_track_conf = 0.5  # minimum tracking confidence in %. Exp: 0.5 are 50%

    # Calculate angle

    # Detects and tracks the positions of landmarks.
    # It needs a video streem as a parameter
    def detection_and_tracking(self, cap):
        # Setup mediapipe instance
        with self.mp_pose.Pose(min_detection_confidence=self.min_det_conf,
                               min_tracking_confidence=self.min_track_conf) as pose:
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

                # Extract Landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                    for landmark in self.mp_pose.PoseLandmark:
                        print(landmark)
                    print(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value])
                except:
                    pass

                shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # print(results)
                # print(results.pose_landmarks)
                # print(self.mp_pose.POSE_CONNECTIONS)

                # Render detections
                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                               self.mp_drawing.DrawingSpec(color=self.color_dots, thickness=2,
                                                                           circle_radius=2),
                                               # landmark drawing specs(dots)
                                               self.mp_drawing.DrawingSpec(color=self.color_lines, thickness=2,
                                                                           circle_radius=2)
                                               # connection drawing specs (lines)
                                               )

                flipped_video = cv2.flip(image, 1)  # mirror image of the feed
                cv2.imshow("Mediapipe Feed",
                           flipped_video)  # This gives as a popup on our screen with the visualisation.
                # We pass the Title and the image

                if cv2.waitKey(10) & 0xFF == ord("q"):  # Listener for the quit button. in this case is the letter 'q'
                    break
            cap.release()  # It releases the capture device
            cv2.destroyWindow()  # and destroys the window

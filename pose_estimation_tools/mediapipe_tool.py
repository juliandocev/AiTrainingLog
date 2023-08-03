import cv2
import numpy as np
from mediapipe import *
from utils import calculations as calc

MP_POSE = solutions.pose
MP_DRAWING = solutions.drawing_utils
COLOR_DOTS = (245, 117, 66)
COLOR_LINES = (245, 66, 280)
MIN_DET_CONF = 0.5  # minimum detection confidence in %. Exp: 0.5 are 50%
MIN_TRACK_CONF = 0.5  # minimum tracking confidence in %. Exp: 0.5 are 50%


# Calculate angle

# Detects and tracks the positions of landmarks.
# It needs a video streem as a parameter
def detection_and_tracking():
    # Curl counter variables
    curl_counter = 0
    stage = None  # the stage represents if we ar at the downside or upside of the curl

    # Get the video feed from the camera and make detections
    cap = cv2.VideoCapture(0)  # We are grabbing or setting the vide capture device. The 0 represent the webcam

    # Setup mediapipe instance
    with MP_POSE.Pose(min_detection_confidence=MIN_DET_CONF,
                      min_tracking_confidence=MIN_TRACK_CONF) as pose:
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

                # Get coordinates
                left_shoulder = [landmarks[MP_POSE.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[MP_POSE.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[MP_POSE.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[MP_POSE.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[MP_POSE.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[MP_POSE.PoseLandmark.LEFT_WRIST.value].y]

                left_hip = [landmarks[MP_POSE.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[MP_POSE.PoseLandmark.LEFT_HIP.value].y]

                # Calculate angle
                left_elbow_angle = calc.calculate_angle(left_shoulder, left_elbow, left_wrist)

                left_shoulder_angle = calc.calculate_angle(left_hip, left_shoulder,
                                                           left_elbow)

                # Visualize angle
                cv2.putText(img=image, text=str(left_elbow_angle),
                            org=calc.joint_position_on_screen(left_elbow),  # Visualize this is calculation of
                            # the positioning of the text in respect of the feed size tuple(np.multiply(elbow,
                            # [640, 480]).astype(int))
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=2,
                            lineType=cv2.LINE_AA
                            )

                # Curl counter logic
                if left_elbow_angle > 160:
                    stage = "down"
                if left_elbow_angle < 30 and stage == "down":
                    stage = "up"
                    curl_counter += 1
                    print(curl_counter)

            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0, 0), (255, 73), (255, 117, 16), -1)

            # Rep data
            cv2.putText(image, "Reps", (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(curl_counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2,
                        cv2.LINE_AA)

            # Stage data
            cv2.putText(image, "Stage", (65, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Render detections
            MP_DRAWING.draw_landmarks(image, results.pose_landmarks, MP_POSE.POSE_CONNECTIONS,
                                      MP_DRAWING.DrawingSpec(color=COLOR_LINES, thickness=2,
                                                             circle_radius=2),
                                      # landmark drawing specs(dots)
                                      MP_DRAWING.DrawingSpec(color=COLOR_LINES, thickness=2,
                                                             circle_radius=2)
                                      # connection drawing specs (lines)
                                      )

            # flipped_video = cv2.flip(image, 1)  # mirror image of the feed
            cv2.imshow("Mediapipe Feed",
                       image)  # This gives as a popup on our screen with the visualisation.
            # We pass the Title and the image

            if cv2.waitKey(10) & 0xFF == ord("q"):  # Listener for the quit button. in this case is the letter 'q'
                break
        cap.release()  # It releases the capture device
        cv2.destroyAllWindows()  # and destroys all windows

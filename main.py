import cv2
from pose_estimation_tools import mediapipe_tool


# Get the video feed from the camera and make detections
cap = cv2.VideoCapture(0)  # We are grabbing or setting the vide capture device. The 0 represent the webcam

mp_tool = mediapipe_tool.MediaPipeTool()

mp_tool.detection_and_tracking(cap)



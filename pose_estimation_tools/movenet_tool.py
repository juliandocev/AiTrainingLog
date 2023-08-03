import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}
COLOR_DOTS = (0, 255, 0)
COLOR_CONNECTIONS = (0, 0, 255)
CONFIDENCE_THRESHOLD = 0.4
# Load TFLite model and allocate tensors.
TF_MODEL_LIGHTNING = "lite-model_movenet_singlepose_lightning_3.tflite"  # A frame of video or an image, represented as a float32 tensor of shape: 192x192x3.
SIZE_IMG_LIGHTNING = 192
# Channels order: RGB with values in [0, 255].
TF_MODEL_THUNDER = "lite-model_movenet_singlepose_thunder_3.tflite"  # A frame of video or an image, represented as a float32 tensor of shape: 256x256x3.
SIZE_ING_THUNDER = 256


# Channels order: RGB with values in [0, 255].


def detection_and_tracking():
    tf_model = TF_MODEL_LIGHTNING

    interpreter = tf.lite.Interpreter(model_path=tf_model)
    interpreter.allocate_tensors()

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()

        # Reshape image
        # We copy the last frame
        img = frame.copy()
        # And we resize it following the documentation:
        # A frame of video or an image, represented as a float32 tensor of shape: 192x192x3.
        # Channels order: RGB with values in [0, 255].
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), SIZE_IMG_LIGHTNING, SIZE_IMG_LIGHTNING)
        input_image = tf.cast(img, dtype=tf.float32)

        # Setup input and output
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Make predictions
        interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
        # print(keypoints_with_scores)

        # right_eye = keypoints_with_scores[0][0][2]
        # left_elbow = keypoints_with_scores[0][0][17]

        # Rendering
        draw_connections(frame, keypoints_with_scores)

        draw_keypoints(frame, keypoints_with_scores)

        cv2.imshow("MoveNet Lightning", frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):  # Listener for the quit button. in this case is the letter 'q'
            break

    cap.release()
    cv2.destroyAllWindows()


def draw_keypoints(frame, keypoints, ):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))  # we take the keypoints coordinates and me multiply by
    # the frame shape. The c is the confidence. We don't want to change it that's why we multiply by 1

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf >= CONFIDENCE_THRESHOLD:
            cv2.circle(frame, (int(kx), int(ky)), 4, COLOR_DOTS, -1)


def draw_connections(frame, keypoints):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in EDGES.items():
        p1, p2 = edge  # we take the joint pairs
        y1, x1, c1 = shaped[p1]  # we grab the coordinates for those specific joints
        y2, x2, c2 = shaped[p2]  # we grab the coordinates for those specific joints

        if (c1 > CONFIDENCE_THRESHOLD) & (c2 > CONFIDENCE_THRESHOLD):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), COLOR_CONNECTIONS, 2)

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2


#
class MoveNetTool:

    def detection_and_tracking(self):
        # Load TFLite model and allocate tensors.
        tf_model = "lite-model_movenet_singlepose_lightning_3.tflite"

        interpreter = tf.lite.Interpreter(model_path=tf_model)
        interpreter.allocate_tensors()

        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()

            # Reshape image

            # Setup input and output

            # Make predictions





            cv2.imshow("MoveNet Lightning", frame)

            while cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # with open(self.tflite_model_file, 'rb') as fid:
        #     tflite_model = fid.read()
        #
        # interpreter = tf.lite.Interpreter(model_content=tflite_model)
        # interpreter.allocate_tensors()
        #
        # input_index = interpreter.get_input_details()[0]["index"]
        # output_index = interpreter.get_output_details()[0]["index"]
        #
        # # Gather results for the randomly sampled test images
        # predictions = []
        #
        # test_labels, test_imgs = [], []
        # for img, label in tqdm(test_batches.take(10)):
        #     interpreter.set_tensor(input_index, img)
        #     interpreter.invoke()
        #     predictions.append(interpreter.get_tensor(output_index))
        #
        #     test_labels.append(label.numpy()[0])
        #     test_imgs.append(img)

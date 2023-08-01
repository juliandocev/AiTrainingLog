import numpy as np


# Calculates an angle degree from 3 points
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


# Calculates the position of the joint in respect of the feed size
def joint_position_on_screen(joint):
    return tuple(np.multiply(joint, [640, 480]).astype(int))

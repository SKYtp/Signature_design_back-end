import cv2
import numpy as np
import math

import cv2
import numpy as np

def extract_all_contour_points(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found or cannot be loaded: {image_path}")

    # Invert the image colors
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    if binary_image is None:
        raise ValueError("Binary image could not be created.")

    # Find all contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contours found in the image.")

    # Collect all points from all contours
    all_points = []
    for contour in contours:
        points = contour.reshape(-1, 2)
        all_points.extend(points)

    # Convert the list of points to a numpy array
    all_points = np.array(all_points)
    if all_points.size == 0:
        raise ValueError("No points were found in contours.")

    # Flip the y-coordinates
    max_y = image.shape[0] - 1
    all_points[:, 1] = max_y - all_points[:, 1]

    return all_points


def group_cal(points):
    group_point = [[]]
    group_num = 0
    for i in range(len(points) - 1):
        group_point[group_num].append([points[i][0], points[i][1]])
        if (abs(points[i][0] - points[i+1][0]) >= 4) or (abs(points[i][1] - points[i+1][1]) >= 4):
            group_num += 1
            group_point.append([])
    group_point[group_num].append([points[-1][0], points[-1][1]])
    # group_point = np.array(group_point)
    return group_point

def compare_two_group(points, group_num):
    group1 = points[group_num[0]]
    group2 = points[group_num[1]]
    # return [First Char of Signature, Tail of Signature]
    if(group1[0][0] <= group2[0][0]):
        return [group_num[0], group_num[1]]
    else:
        return [group_num[1], group_num[0]]

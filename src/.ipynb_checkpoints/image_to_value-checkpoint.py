from get_contour import *
from distance_tall import distance_and_tall_of_two_group
from angle_of_signature import angle_of_signature
from cross_check import get_cross
import cv2

def final_value_contour_allSig(input_image):
    # return distance, tall, angle
    image = input_image
    all_points = extract_all_contour_points(image)
    group_points = group_cal(all_points)
    compare = compare_two_group(group_points, biggest_two_num(group_points))
    h_l_n = highest_lowest_nearest(group_points, compare)
    # print('Distance and Tall :', distance_and_tall_of_two_group(h_l_n))
    # print('Angle of signature : ' + str(angle_of_signature(h_l_n)) + '°')
    tmp = distance_and_tall_of_two_group(h_l_n)
    result = [tmp[0], tmp[1], angle_of_signature(h_l_n)]
    return result
    
def final_value_contour_head(input_image):
    # return line_connection, cross_line_number
    image = input_image
    all_points = extract_all_contour_points(image)
    group_points = group_cal(all_points)
    cross_num = get_cross(cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB))
    # print("cross_num: " + str(cross_num))
    # print('Line dis : ' + str(bool(len(group_points) != 1)))
    result = [bool(len(group_points) != 1), cross_num]
    return result

image = cv2.imread('./test_pic/21.png', cv2.IMREAD_GRAYSCALE)
if image is None:
        raise ValueError(f"Image not found or cannot be loaded")

image2 = cv2.imread('./test_pic/6.png', cv2.IMREAD_GRAYSCALE)
if image2 is None:
        raise ValueError(f"Image not found or cannot be loaded")
# # Load the image using OpenCV
# image2_yolo = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel RGB

asign = final_value_contour_allSig(image)
hsign = final_value_contour_head(image2)
print('--------------------------------------')
print(asign)
print(hsign)
print(image)


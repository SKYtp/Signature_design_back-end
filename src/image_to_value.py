from get_contour import *
from distance_tall import distance_and_tall_of_two_group
from angle_of_signature import angle_of_signature
from cross_check import get_cross

def final_value_contour_allSig(input_image_path):
    # return distance, tall, angle
    image_path = input_image_path
    all_points = extract_all_contour_points(image_path)
    group_points = group_cal(all_points)
    compare = compare_two_group(group_points, biggest_two_num(group_points))
    h_l_n = highest_lowest_nearest(group_points, compare)
    # print('Distance and Tall :', distance_and_tall_of_two_group(h_l_n))
    # print('Angle of signature : ' + str(angle_of_signature(h_l_n)) + '°')
    tmp = distance_and_tall_of_two_group(h_l_n)
    result = [tmp[0], tmp[1], angle_of_signature(h_l_n)]
    return result
    
def final_value_contour_head(input_image_path):
    # return line_connection, cross_line_number
    image_path = input_image_path
    all_points = extract_all_contour_points(image_path)
    group_points = group_cal(all_points)
    cross_num = get_cross(image_path)
    # print("cross_num: " + str(cross_num))
    # print('Line dis : ' + str(bool(len(group_points) != 1)))
    result = [bool(len(group_points) != 1), cross_num]
    return result

asign = final_value_contour_allSig('./test_pic/21.png');
hsign = final_value_contour_head('./test_pic/6.png')
print('--------------------------------------')
print(asign)
print(hsign)


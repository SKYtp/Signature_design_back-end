def distance_and_tall_of_two_group(points):
    return [abs(points[0][0] - points[1][0]) / abs((points[3][1] - points[5][1])), (points[3][1] - points[5][1]) / (points[2][1] - points[4][1])]
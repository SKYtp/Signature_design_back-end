import math

def angle_of_signature(points):
    return round(math.degrees(math.atan(abs(points[5][1] - points[4][1]) / abs(points[5][0] - points[4][0]))), 2)
    # return round(math.degrees(math.atan(abs(points[5][1] - points[4][1]) / abs(points[5][0] - points[4][0]))))
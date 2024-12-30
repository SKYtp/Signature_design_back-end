def distance_and_tall_of_two_group(points):
    # return float(abs(points[0][0] - points[1][0]) / abs((points[3][1] - points[5][1]))), float((points[3][1] - points[5][1]) / (points[2][1] - points[4][1]))
    # Perform calculations
    distance = abs(points[0][0] - points[1][0]) / abs((points[3][1] - points[5][1]))
    tall = (points[3][1] - points[5][1]) / (points[2][1] - points[4][1])
    # Format to 2 decimal places and convert to Python floats
    return [float(round(distance, 2)), float(round(tall, 2))]
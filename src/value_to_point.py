import numpy as np

def value_2_point1(x): # ตำแหน่งประธานต้องอยู่ในระนาบเดียวกับตำแหน่งบริวาร
    result = np.round(1 - 0.000256 * (x - 0)**4, 5)
    if(result > 1):
        return 1
    elif(result > 0):
        return result
    else:
        return 0
    # return np.round(1 - 0.000256 * (x - 0)**4, 5)

def value_2_point2_3(x): # ความสูงบริวารต้องเป็นเศษหนึ่งส่วนสองของความสูงประธาน, ประธานกับบริวารต้องเว้นว่างเป็นเศษหนึ่งส่วนสองของความสูงบริวาร
    result = np.round(1 - 1600 * (x - 0.5)**4, 5)
    if(result > 1):
        return 1
    elif(result > 0):
        return result
    else:
        return 0
    # return np.round(1 - 1600 * (x - 0.5)**4, 5)

def value_2_point4(x): # ตัวอักษรในลายเซ็นจะต้องไม่มีการขาดของเส้นภายในตัวอักษร
    if(x == False):
        return 1
    else:
        return 0
    
def value_2_point5(x, b): # ประธานต้องไม่มีเส้นตัดกันที่เกิดจากการเซ็น
    if(x == 0 and (b == "ป" or b == "ว")):
        return 1
    elif(x == 1 and b == "ส"):
        return 1
    else:
        return 0




print(value_2_point2_3(0.45))
print(value_2_point4(False))
print(value_2_point5(0, "ว"))
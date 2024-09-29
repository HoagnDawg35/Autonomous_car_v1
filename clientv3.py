from client_lib import GetStatus,GetRaw,GetSeg,AVControl,CloseSocket
import cv2
import numpy as np
import math
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

CHECKPOINT_1 = 60
CHECKPOINT_2 = 40 #Mượt nhất = 30
CHECKPOINT_0 = 70

# CHECKPOINT_3 = 
# CHECKPOINT_4 = 

pre_t = time.time()
err_arr = np.zeros(5)

Kp_vals = []
Ki_vals = []
Kd_vals = []
time_vals = []


# -----------------------------------------------------------------------------
def vector_length(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def AngCal(image):
    global lane_width_2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = (gray * (255 / np.max(gray))).astype(np.uint8)
    _, bin = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    h, w = bin.shape

    line_row_0 = bin[CHECKPOINT_0, :]
    line_row_1 = bin[CHECKPOINT_1, :]  # Take one row from the image to check
    line_row_2 = bin[CHECKPOINT_2, :]

    flag = True
    min_0 = 0
    max_0 = 0

    #Tìm pixel trắng (255) đầu tiên và cuối cùng trong hàng
    for x, y in enumerate(line_row_0):
        if y == 255 and flag:
            flag = False
            min_0 = x
        elif y == 255:
            max_0 = x

    #Min tọa độ của min_1/max_1 và min_2/max_2 
    flag = True
    min_1 = 0
    max_1 = 0

    #Tìm pixel trắng (255) đầu tiên và cuối cùng trong hàng
    for x, y in enumerate(line_row_1):
        if y == 255 and flag:
            flag = False
            min_1 = x + 20
        elif y == 255:
            max_1 = x - 20

    flag = True
    min_2 = 0
    max_2 = 0

    #Tìm pixel trắng (255) đầu tiên và cuối cùng trong hàng
    for x, y in enumerate(line_row_2):
        if y == 255 and flag:
            flag = False
            min_2 = x
        elif y == 255:
            max_2 = x


    #----------------------------------------------------------------------------------
    
    #Hiển thị ảnh 
    cv2.imshow("Binary", bin)

    # tính center 1 và center 2
    center_0 = (min_0 + max_0) // 2
    center_1 = (min_1 + max_1) // 2
    center_2 = (min_2 + max_2) // 2


    # Tính khoảng cách giữa 2 điểm trong 1 hàng Checkpoint
    lane_width_1 = max_1 - min_1
    lane_width_2 = max_2 - min_2

    # Phát hiện ngã 3
    length_min = vector_length(min_1, CHECKPOINT_1, min_2, CHECKPOINT_2)
    length_max = vector_length(max_1, CHECKPOINT_1, max_2, CHECKPOINT_2)
    
    # Tính khoảng cách giữa max_1/max_2 và min_1/min_2
    # print(f"Độ dài vecter trái: ", length_min)
    # print(f"Độ dài vecter phải: ", length_max)

    # Ngã ba
    Right_cross = False
    Left_cross = False
    if min_1 > 20 and max_1 == 299 and (lane_width_1)**2 > 62000: #Phát hiện ngã rẽ bên Trái | 52900
        print(f"Right Cross")
        Left_cross = True
    elif min_1 == 20 and max_1 < 299 and (lane_width_1)**2 > 62000: #Phát hiện ngã rẽ bên Phải | 52900
        print(f"Left Cross")
        Right_cross = True
    else:
        Left_cross = False; Right_cross = False

    if Right_cross == True:
        error = min_1 + 120
    elif Left_cross == True: 
        error = max_1 - 120
    else: 
        error = center_1 - w // 2

    # Ngã tư
    if lane_width_2 == 319: #Nếu có ngã tư thì đổi error thành tâm của center 2 không thì giữ nguyên
        Right_cross = False; Left_cross = False
        print(f"Intersection detected")
        error = center_2 - w // 2 
    else:                      
        Right_cross = False; Left_cross = False   
        print(f"No intersection detected")
        error = center_1 - w // 2
    
    # Hiển thị các điểm Max, Min, Center
    cv2.circle(crop_seg, (min_1, CHECKPOINT_1), radius=4, color = (255, 255, 0), thickness=-1)
    cv2.circle(crop_seg, (max_1, CHECKPOINT_1), radius=4, color = (255, 255, 0), thickness=-1)
    cv2.circle(crop_seg, (center_1, CHECKPOINT_1), radius=4, color = (255, 0, 0), thickness = -1)

    cv2.circle(crop_seg, (min_2, CHECKPOINT_2), radius=4, color = (255, 255, 0), thickness=-1)
    cv2.circle(crop_seg, (max_2, CHECKPOINT_2), radius=4, color = (255, 255, 0), thickness=-1)
    cv2.circle(crop_seg, (center_2, CHECKPOINT_2), radius=4, color = (255, 0, 0), thickness = -1)
    """
    cv2.circle(crop_seg, (min_I, CHECKPOINT_I), radius=4, color = (255, 255, 0), thickness=-1)
    cv2.circle(crop_seg, (max_I, CHECKPOINT_I), radius=4, color = (255, 255, 0), thickness=-1)
    cv2.circle(crop_seg, (center_I, CHECKPOINT_I), radius=4, color = (255, 0, 0), thickness = -1)
    """
    # Vẽ đường thẳng min_1/min_2 và max_1/max_2
    cv2.line(crop_seg, (min_1, CHECKPOINT_1), (min_2, CHECKPOINT_2), (255, 255, 0), thickness = 1)
    cv2.line(crop_seg, (max_1, CHECKPOINT_1), (max_2, CHECKPOINT_2), (255, 255, 0), thickness = 1)

    # #Vẽ line chia bố cục
    # cv2.line(segment_image, (CHECKPOINT_3, 0), (CHECKPOINT_3, 180), (0, 255, 0), thickness = 1)
    # cv2.line(segment_image, (CHECKPOINT_4, 0), (CHECKPOINT_4, 180), (0, 255, 0), thickness = 1)

    # Hiện Segment
    cv2.imshow("segment_image", crop_seg)

    # Hiện thông số Max và Min
    print(f"Min 1: {min_1} | Max_1: {max_1} | Min_2: {min_2} | Max_2: {max_2}")
    
    print(f"error: {error} | Center_1: {center_1}")
    return error

#----------------------------------------------------------------------------------
def PID(err, Kp, Ki, Kd):
    global pre_t
    err_arr[1:] = err_arr[0:-1]
    err_arr[0] = err
    delta_t = time.time() - pre_t
    pre_t = time.time()

    P = Kp*err
    D = Kd*(err - err_arr[1])/delta_t
    I = Ki*np.sum(err_arr)*delta_t
    angle = P + I + D

    if abs(angle) > 25:
        angle = np.sign(angle)*25
        
    return int(angle), Kp, Ki, Kd

#----------------------------------------------------------------------------------
# def update_plot(frame):
#     # plt.cla()
#     # plt.plot(time_vals, Kp_vals, label="Kp")
#     # plt.plot(time_vals, Ki_vals, label="Ki")
#     # plt.plot(time_vals, Kd_vals, label="Kd")

#     # plt.xlabel('Time (s)')
#     # plt.ylabel('PID values')
#     # plt.title('Real-time PID values')
#     # plt.legend()
#     # plt.tight_layout()

#----------------------------------------------------------------------------------
# fig = plt.figure()
# ani = FuncAnimation(fig, update_plot, interval=100)  # Update every 100ms

"""
Speed max = 80
speed min = 30
Kp = 0.11
Kd = 0.035
Ki = 0.0
"""

if __name__ == "__main__":
    Kp, Ki, Kd = 0.18, 0.0, 0.065# Hệ số PID ban đầu
    speed = 30 #Tốc độ ban đầu
    speed_max = 60  # Tốc độ tối đa | Mượt nhất = 50
    speed_min = 20  # Tốc độ tối thiểu | Mượt nhất = 20
    color = (0, 255, 0)
    start_time = time.time()
    try:
        while True:
            # Lấy ảnh segment
            state = GetStatus()
            segment_image = GetSeg()  # Lấy ảnh phân đoạn
        
            #Lấy thông số của ảnh
            height, width, _ = segment_image.shape
            crop_seg = segment_image[70:height, 0:width]
            
            #Calculate the error from the image
            error = AngCal(crop_seg)  
            
            # Tính toán góc lái từ Adaptive PID control
            angle, Kp, Ki, Kd = PID(error, Kp, Ki, Kd)
            
            # if (angle <= 25 and angle >= 15)or (angle >= -25 and angle <= -15):
            #     print(f"The car is taking Hard-turn")
            # elif (angle < 15 and angle >= 5)or (angle > -15 and angle <= -5):
            #     print(f"The car is taking turn")
            # elif (angle < 5 and angle > 0)or (angle > 5 and angle < 0):
            #     print(f"The car is taking a slightly-turn")
            # else: 
            #     print(f"The car is driving straight")
            
            # Điều khiển xe dựa trên góc tính được
            AVControl(speed=speed, angle=angle)

            # Điều chỉnh tốc độ dựa trên hàm tuyến tính y = -A * error + B
            A = 0.8 #Mượt nhất = 0.5
            speed = -A * abs(error) + speed_max

            # Giới hạn tốc độ trong khoảng [speed_min, speed_max]
            speed = np.clip(speed, speed_min, speed_max)
            
            #In góc lái và speed
            # print(f"Angle: {angle} |  Speed: {speed}")
            # print(f"Kp: {Kp:2f}, Ki: {Ki:2f}, Kd: {Kd:2f}")
            print(f"-----------------------------")

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    finally:
        print('closing socket')
        CloseSocket()

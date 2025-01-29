import cv2
import numpy as np
from ultralytics import YOLO


vehicle_positions_y = {}           #variables define that are used
vehicle_positions_x = {}
crossed_vehicles_up = set()
crossed_vehicles_down = set()
vehicle_count_up = 0
vehicle_count_down = 0
REFERENCE_DISTANCE = 10                 # assuming reference distance
FPS = 30                        # my video is at 30 fps
LINE_Y = 300  # can be calculated as horizontal pixels with floor devision of 2


def process_frame(frame, model):
    global vehicle_count_up, vehicle_count_down  # taking from the defined variable
    results = model(frame)   #apply yolo
    
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)  
            center_x = (x1 + x2) // 2  
            center_y = (y1 + y2) // 2 
            vehicle_name = model.names[int(cls)]  
            vehicle_no = match_vehicle(center_y, center_x)
            
            prev_y = vehicle_positions_y.get(vehicle_no, center_y)
            prev_x = vehicle_positions_x.get(vehicle_no, center_x)  
            speed = estimate_speed(prev_x,center_x,prev_y, center_y, FPS, REFERENCE_DISTANCE)
            cv2.putText(frame, f"{vehicle_name} Speed: {speed:.2f} km/h", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            vehicle_positions_y[vehicle_no] = center_y  #saving for previous y
            vehicle_positions_x[vehicle_no] = center_x
            
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{vehicle_name}", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # checking of vehicle crossing
            if vehicle_no not in crossed_vehicles_down and prev_y < LINE_Y and center_y >= LINE_Y:
                vehicle_count_down += 1
                crossed_vehicles_down.add(vehicle_no)
            elif vehicle_no not in crossed_vehicles_up and prev_y > LINE_Y and center_y <= LINE_Y:
                vehicle_count_up += 1
                crossed_vehicles_up.add(vehicle_no)
    
    draw_overlay(frame)
    return frame

def estimate_speed(x1,x2,y1, y2, frame_rate, ref_distance):       #calculating speed of object
    pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
    speed = (pixel_distance / frame_rate) * ref_distance  
    return speed*3.6

def match_vehicle(center_y, center_x):
    vehicle_no = None
    for vno, prev_y in vehicle_positions_y.items():
        if abs(prev_y - center_y) < 50:  #here assumption made
            vehicle_no = vno
            break
    
    if vehicle_no is None:
        vehicle_no = len(vehicle_positions_y) + 1  
    
    return vehicle_no

def draw_overlay(frame):
    cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0, 0, 255), 2)
    cv2.putText(frame, f"Upward Count: {vehicle_count_up}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Downward Count: {vehicle_count_down}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)



model = YOLO('yolov8n.pt')
video = cv2.VideoCapture('video/1171461-hd_1920_1080_30fps.mp4')

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break 
    
    frame = process_frame(frame, model)
    cv2.imshow("Vehicle Detection", frame)  
    
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break  # by pressing e you can quit

video.release()
cv2.destroyAllWindows()


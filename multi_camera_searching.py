from random import randint
import cv2
import numpy as np
import time
import signal
from pathlib import Path
import os
import math
from tracking.detector import YOLOv9
from utils.logger import Logger
from utils.config import Config
import datetime
from tracking.sct import SCTThread
from tracking.frame_manager import FramedataManager
from tracking.mct import MCT
from server.server_thread import ServerTask
from tracking.utils import get_current_utc_iso
logger = Logger()
cfg = Config()

stopped = False
processing = False
list_cameras = dict()
list_frame_manager = dict()
global_id = 0
server= None

# Define the signal handler function
def signal_handler(sig, frame):
    logger.info("Ctrl + C pressed. Cleaning up and exiting...")
    stopped = True
    
    while processing:
        print("waiiting....")
        time.sleep(1)
    logger.info("Waiting main thread done!!!")
    list_frame_manager.clear()
    for cam in list_cameras:
        list_cameras[cam].stop(by_SIGINT=True)
        if list_cameras[cam].is_alive():
            list_cameras[cam].join(timeout=1)  # Wait for 2 seconds
    list_cameras.clear()
    cv2.destroyAllWindows()
    server.stop()
    logger.info("Stopped all threads")
    exit(0)  # Exit the program

# Register the signal handler for SIGINT (Ctrl + C)
signal.signal(signal.SIGINT, signal_handler)

def remove_oldest_file(dir, ext, rotate_size=5):
    files = [f for f in Path(dir).glob(ext) if f.is_file()]
    if len(files) > rotate_size:
        files.sort(key=lambda x: x.stat().st_mtime)
        # Remove the oldest file
        oldest_file = files[0]
        oldest_file.unlink()  # Deletes the file

def save_person_image(output_dir, camid, frameid, pid, image, rotate_size=5):
    if image.size == 0:
        return
    dir = os.path.join(output_dir, f'cam{camid}',f'p{pid}')
    os.makedirs(dir, exist_ok=True)
    remove_oldest_file(dir, '*.jpg',rotate_size)
    path = os.path.join(dir, f'f{frameid}.jpg')
    cv2.imwrite(path, image)

def draw_boxes(frame, ids, bboxes, color_list):
    for id, bbox in zip(ids, bboxes):
        x1, y1, x2, y2 = [int(x) for x in bbox]
        # x1, y1, x2, y2 = bbox
        color = color_list[id]
        label = f'ID:{id}'
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color,1)
        cv2.rectangle(frame, (x1, y1), (x2, y1 + h), color, -1)
        cv2.putText(frame, label, (x1, y1+h), cv2.FONT_HERSHEY_SIMPLEX, 2.0, [255, 255, 255], 2)

def draw_tracks(frame, ids,  tracks, color_list):
    for id, track in zip(ids,tracks):     
        for i,_ in  enumerate(track): 
            if i < len(track)-1:
                cv2.line(frame, (int(track[i][0]),
                    int(track[i][1])), 
                    (int(track[i+1][0]),
                    int(track[i+1][1])),
                    color_list[id], 2)
def main():
    start_time = get_current_utc_iso()
    logger.info("=======================================")
    logger.info("=========Multi-Camera Searching=========")
    logger.info(f"========= {start_time} =========")
    logger.info("=======================================")
    
    # sources = ["./data/cameras/s2-c0.avi", "./data/cameras/s2-c1.avi"]
    # sources = ["./data/videos/4p-c0.avi", "./data/videos/4p-c1.avi"]
    # sources = ["./data/videos/4p-c0.avi", "./data/videos/4p-c1.avi", "./data/videos/4p-c2.avi"]
    # sources = ["D:/projects/ReID_Survey/test_reid/t3/c1.avi",
    #            "D:/projects/ReID_Survey/test_reid/t3/c2.avi",
    #            "D:/projects/ReID_Survey/test_reid/t3/c3.avi",
    #            "D:/projects/ReID_Survey/test_reid/t3/c4.avi",]
    sources = ["D:/projects/ReID_Survey/test_reid/t4/c1.avi",
               "D:/projects/ReID_Survey/test_reid/t4/c2.avi"
               ]

    OUTPUT_DIR = "./results/"

    for id, src in enumerate(sources):
        camid = f'cam{id}'
        list_frame_manager[camid] = FramedataManager()
        list_cameras[camid] = SCTThread(camid, src, list_frame_manager[camid])

    for cam in list_cameras:
        list_cameras[cam].daemon = True
        list_cameras[cam].start()
    
    # start server
    global server
    server = ServerTask(list_cameras)
    server.start()

    grid_view_cols = 2
    grid_view_rows = 1
    output_view_height = 960
    output_view_width = 1280
    window_y = int(output_view_height/grid_view_rows)
    window_x = int(output_view_width/grid_view_cols)
    visual_frame = np.zeros((output_view_height, output_view_width, 3), dtype=np.uint8)
    
   
    # Below VideoWriter object will create 
    # a frame of above defined The output  
    # is stored in 'filename.avi' file. 
    # Get current time
    now = datetime.datetime.now()
    # Format the time as a string
    formatted_time =  time.strftime("%Y%m%d-%H%M%S")
    size = (output_view_width, output_view_height) 
    output_video = f'{OUTPUT_DIR}/videos/mct_{formatted_time}.mp4'
    logger.info(f"Output video at: {output_video}" )
    writer = cv2.VideoWriter(output_video,  
                            cv2.VideoWriter_fourcc(*'XVID'), 
                            10, size) # *'XVID', 'mp4v'
    
    rand_color_list = []
    for i in range(0,5005):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)
    have_frame = False
    try:
        while not stopped:
            processing= True
            for camid in list_frame_manager:
                cam_idx = int(camid[3:])
                framedata = list_frame_manager[camid].pop_framedata()
                if framedata is not None:
                    have_frame = True
                    vis_window = cv2.resize(framedata.vis_image, (window_x, window_y))
                    cv2.putText(vis_window,f'{camid}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                    x = int(cam_idx  % grid_view_cols)
                    y = int(cam_idx / grid_view_cols)
                    visual_frame[y * window_y : window_y + y * window_y, x * window_x: window_x + x * window_x] = vis_window
            size_w = output_view_width / grid_view_cols
            for i in range(1, grid_view_cols):
                cv2.line(visual_frame,(int(i * size_w),0),(int(i * size_w),output_view_height),(0,0,255),2)
            size_h = output_view_height / grid_view_rows
            for i in range(1, grid_view_rows):
                cv2.line(visual_frame,(0, int(size_h * i)),(output_view_width, int(size_h * i)),(0,0,255),2)
            if have_frame:
                writer.write(visual_frame)
                cv2.imshow(f"Multi-Camera Tracking",visual_frame)
            if cv2.waitKey(25) == 'q':
                    break
            processing = False

    except Exception as e:
        logger.error( "main: Unexpected error %s " % e )
        raise
    except (KeyboardInterrupt, SystemExit):
        logger.info(" Program is exiting by SIGINT signal...")
    writer.release() 
    logger.info(f"Saved output video at: {output_video}")

if __name__ == "__main__":
    main()
    logger.info("Ended App!!!")


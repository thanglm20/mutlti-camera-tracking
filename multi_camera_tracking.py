from random import randint
import cv2
import numpy as np
import time
import signal
from pathlib import Path
import os
from tracking.detector import YOLOv9
from utils.logger import Logger
from tracking.sct import SCTThread
from tracking.frame_manager import FramedataManager
from tracking.mct import MCT
logger = Logger()

OUTPUT_DIR = "./outputs"
stopped = False
processing = False
list_cameras = dict()
list_frame_manager = dict()
global_id = 0
threshold = 1.0

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
        list_cameras[cam].join(timeout=1)  # Wait for 2 seconds
    list_cameras.clear()
    cv2.destroyAllWindows()
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
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color,1)
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + 10), color, -1)
        cv2.putText(frame, label, (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255, 255, 255], 1)

def draw_tracks(frame, ids,  tracks, color_list):
    max_len = 60
    for id, track in zip(ids,tracks):     
        for i,_ in  enumerate(track): 
            if i < len(track)-1 and i >= len(track) - max_len:
                cv2.line(frame, (int(track[i][0]),
                    int(track[i][1])), 
                    (int(track[i+1][0]),
                    int(track[i+1][1])),
                    color_list[id], thickness=3)
def main():
    logger.info("=======================================")
    logger.info("=========Multi-Camera Tracking=========")
    logger.info("=======================================")
    
    # sources = ["./data/cameras/s2-c0.avi", "./data/cameras/s2-c1.avi"]
    # sources = ["./data/videos/4p-c0.avi", "./data/videos/4p-c1.avi"]
    sources = ["./data/videos/4p-c0.avi", "./data/videos/4p-c1.avi", "./data/videos/4p-c2.avi"]


    # sources = ["./data/cameras/s2-c0.avi"]
    
    for id, src in enumerate(sources):
        list_frame_manager[id] = FramedataManager()
        list_cameras[id] = SCTThread(id, src, list_frame_manager[id])

    for cam in list_cameras:
        list_cameras[cam].daemon = True
        list_cameras[cam].start()
    grid_view_cols = 2
    grid_view_rows = 2
    output_view_height = 960
    output_view_width = 1280
    window_y = int(output_view_height/grid_view_rows)
    window_x = int(output_view_width/grid_view_cols)
    visual_frame = np.zeros((output_view_height, output_view_width, 3), dtype=np.uint8)
    
    size = (output_view_width, output_view_height) 
   
    # Below VideoWriter object will create 
    # a frame of above defined The output  
    # is stored in 'filename.avi' file. 
    result = cv2.VideoWriter(f'{OUTPUT_DIR}/mct_output.avi',  
                            cv2.VideoWriter_fourcc(*'MJPG'), 
                            10, size) 
    mct = MCT()
    rand_color_list = []
    for i in range(0,5005):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)
    try:
        while not stopped:
            processing= True
            for camid in list_frame_manager:
                framedata = list_frame_manager[camid].pop_framedata()
                if framedata is not None:
                    # reid
                    ids, bboxes, tracks = mct.update(camid, framedata.identities, framedata.boxes, framedata.features)
                    # save output
                    for pid, box in zip(ids, bboxes):
                        x1, y1, x2, y2 = [int(x) for x in box]
                        person_image= framedata.org_image[y1:y2, x1:x2]
                        if len(person_image):
                            save_person_image(OUTPUT_DIR, camid, framedata.frameid, pid, person_image)
                    # visualize
                    draw_boxes(framedata.vis_image, ids, bboxes, rand_color_list)
                    draw_tracks(framedata.vis_image, ids, tracks, rand_color_list)
                    vis_window = cv2.resize(framedata.vis_image, (window_x, window_y))
                    # cv2.putText(vis_window,f'CAM {camid} frame {framedata.frameid}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                    cv2.putText(vis_window,f'CAM {camid}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

                    x = int(camid  % grid_view_cols)
                    y = int(camid / grid_view_cols)
                    visual_frame[y * window_y : window_y + y * window_y, x * window_x: window_x + x * window_x] = vis_window
            for i in range(2, grid_view_cols + 1):
                cv2.line(visual_frame,(int(output_view_width/i),0),(int(output_view_width/i),output_view_height),(0,0,255),2)
            for i in range(2, grid_view_rows + 1):
                cv2.line(visual_frame,(0, int(output_view_height/i)),(output_view_width, int(output_view_height/i)),(0,0,255),2)
            result.write(visual_frame)
            cv2.imshow(f"Multi-Camera Tracking",visual_frame)
            if cv2.waitKey(25) == 'q':
                    break
            processing = False

    except Exception as e:
        logger.error( "main: Unexpected error %s " % e )
        raise
    except (KeyboardInterrupt, SystemExit):
        logger.info(" Program is exiting by SIGINT signal...")
    result.release() 
if __name__ == "__main__":
    main()
    logger.info("Ended App!!!")
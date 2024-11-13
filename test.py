
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


def save_feature(output_dir, camid, frameid, pid, feature, rotate_size=5):
    dir = os.path.join(output_dir, f'cam{camid}',f'p{pid}')
    os.makedirs(dir, exist_ok=True)
    remove_oldest_file(dir, '*.npy', rotate_size)
    path = os.path.join(dir, f'f{frameid}.npy')
    np.save(path, feature)

def find_top_score(mct_data, feature):
    distances = {}
    for pid, person in mct_data.items():
        distance = np.linalg.norm(feature - person["feature"])
        distances[pid] = distance

    pid = min(distances, key=distances.get)
    min_distance = distances[pid]
    return pid, min_distance

def make_new_global_id(camid, frameid, bbox, feature):
    global global_id
    global_id += 1
    return {
        "camid": camid,
        "global_id": global_id,
        "frameid": frameid,
        "feature": feature,
        "bbox": bbox,
        "color": np.random.randint(0, 255, size=3),
    }

def draw_boxes( mct_data, list_pid, frame, frameid):
    for pid in list_pid:
        if mct_data[pid]["frameid"] == frameid:
            x1, y1, x2, y2 = [int(i) for i in mct_data[pid]["bbox"]]
            id = pid
            label = f'G_ID:{id}'
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 253),1)
            cv2.rectangle(frame, (x1, y1 - 10), (x1 + w, y1), (255,144,30), -1)
            cv2.putText(frame, label, (x1, y1),cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255, 255, 255], 1)

def main():
    logger.info("=======================================")
    logger.info("=========Multiple Camera Tracking======")
    logger.info("=======================================")
    
    sources = ["./data/cameras/s2-c0.avi", "./data/cameras/s2-c1.avi"]
    # sources = ["./data/cameras/s2-c0.avi"]
    
    for id, src in enumerate(sources):
        list_frame_manager[id] = FramedataManager()
        list_cameras[id] = SCTThread(id, src, list_frame_manager[id])

    for cam in list_cameras:
        list_cameras[cam].daemon = True
        list_cameras[cam].start()
    grid_view_cols = 2
    grid_view_rows = 1
    output_view_height = 720
    output_view_width = 1280
    window_y = int(output_view_height/grid_view_rows)
    window_x = int(output_view_width/grid_view_cols)
    visual_frame = np.zeros((output_view_height, output_view_width, 3), dtype=np.uint8)
    
    mct_data = dict()
    '''
    camid: int
    pid: int
    box: array
    feature: array
    
    '''
    try:
        while not stopped:
            processing= True
            list_frame = {}
            actice_pid = {}
            for camid in list_frame_manager:
                framedata = list_frame_manager[camid].pop_framedata()
                actice_pid[camid] = []
                if framedata is not None:
                    # save output
                    for pid, box, feature in zip(framedata.identities, framedata.boxes, framedata.features):
                        pid = int(pid)
                        x1, y1, x2, y2 = [int(x) for x in box]
                        person_image= framedata.org_image[y1:y2, x1:x2]
                        save_person_image(OUTPUT_DIR, camid, framedata.frameid, pid, person_image)
                        # save_feature(OUTPUT_DIR, camid, framedata.frameid, pid, feature)
                        # find top 1 score to match global pid
                        if not mct_data:
                            new_person = make_new_global_id(camid, framedata.frameid, box, feature)
                            mct_data[new_person["global_id"]] = new_person
                            actice_pid[camid].append(new_person["global_id"])
                        else:
                            g_pid, score = find_top_score(mct_data, feature)
                            if score > threshold:
                                new_person = make_new_global_id(camid,framedata.frameid, box, feature)
                                mct_data[new_person["global_id"]] = new_person
                                actice_pid[camid].append(new_person["global_id"])
                            else:
                                mct_data[g_pid]["frameid"] = framedata.frameid
                                mct_data[g_pid]["feature"] = feature
                                mct_data[g_pid]["bbox"] = box

                                actice_pid[camid].append(g_pid)
                        
                    list_frame[camid] = (framedata.frameid, framedata.org_image)
            # visualize
            for camid, frame in list_frame.items():
                frameid = frame[0]
                image = frame[1]
                list_pid = actice_pid[camid]
                draw_boxes(mct_data, list_pid, image, frameid)
                vis_window = cv2.resize(image, (window_x, window_y))
                cv2.putText(vis_window,f'CAM {camid} frame {frameid}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                x = int(camid  % grid_view_cols)
                y = int(camid / grid_view_cols)
                visual_frame[y * window_y : window_y + y * window_y, x * window_x: window_x + x * window_x] = vis_window
            for i in range(2, grid_view_cols + 1):
                cv2.line(visual_frame,(int(output_view_width/i),0),(int(output_view_width/i),output_view_height),(0,0,255),2)
            cv2.imshow(f"Output",visual_frame)
            if cv2.waitKey(25) == 'q':
                    break
            processing = False

    except Exception as e:
        logger.error( "main: Unexpected error %s " % e )
        raise
    except (KeyboardInterrupt, SystemExit):
        logger.info(" Program is exiting by SIGINT signal...")



if __name__ == "__main__":
    main()
    logger.info("Ended App!!!")
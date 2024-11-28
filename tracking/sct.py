from threading import Thread, Lock, Event ,Condition
import time
import numpy as np
import cv2
from utils.logger import Logger
logger = Logger()
from tracking.frame_manager import FramedataManager, FrameData
from tracking.detector import YOLOv9
from utils.config import Config
import sys
import os
import shutil
from tracking.sort_tracker import SortTracker
from tracking.feature_extractor import FeatureExtractor
from db.elastic_search import DBManager
from tracking.utils import save_person_image, read_image, get_current_utc_iso
class SCTThread(Thread):
    def __init__(self, camid, source, framedata_manager:FramedataManager):
        """Create a Processor.

        Args:
            camid:
                Unique ID of the camera.
            m_frame_container:
                Container which contains visualized frames for showing output stream.
        """
        # Create camera processor
        self.camid = camid
        self.source = source
        self.framedata_manager = framedata_manager
        self.frameid = 0
        self.stopped = False
        self.cfg = Config.get_instance()
        self.yolo_gpu = self.cfg.yolo.getboolean("gpu", True)
        self.yolo_model = self.cfg.yolo.getstr("yolo_model", "./models/yolov9-c.onnx")
        self.yolo_labels = self.cfg.yolo.getstr("yolo_labels", "./models/coco.yaml")
        self.reid_gpu = self.cfg.reid.getboolean("gpu", True)
        self.reid_model = self.cfg.reid.getstr("reid_model", "./models/reid.onnx")
        self.visualize= self.cfg.output.getboolean("visualize", True)

        self.output_dir = self.cfg.output.getstr("output_dir", "./outputs")

        self.stop_event = Event()
        self.processing = False
        self.condition = Condition()
        Thread.__init__(self) 

    def stop(self,  by_SIGINT=False):
        self.stopped = True
        self.stop_event.set()
        while self.processing:
            time.sleep(0.01)
        self.framedata_manager = None
        self.cfg = None

    def run(self):
        """Main processing thead
        """
        cap = cv2.VideoCapture(self.source)
        assert cap.isOpened(), f'Failed to open {self.source}'
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan

        device = 'cuda' if self.yolo_gpu else 'cpu'
        self.detector = YOLOv9(model_path=f"{self.yolo_model}",
                        class_mapping_path=self.yolo_labels,
                        original_size=(w, h),
                        device=device)
        time.sleep(3)
        sort_max_age = 5 
        sort_min_hits = 2
        sort_iou_thresh = 0.2
        self.tracker = SortTracker(max_age=sort_max_age,min_hits=sort_min_hits,iou_threshold=sort_iou_thresh)
        device = 'cuda' if self.reid_gpu else 'cpu'
        self.extractor = FeatureExtractor(self.reid_model, device, 0)
        self.cam_db_search = DBManager(camid=self.camid, feature_dims=384,create_index=True)

        self.clear_old_output()

        if self.stopped:
            return
        # Start looping
        while not self.stop_event.is_set() and not self.stopped and cap.isOpened():
            try:    
                self.processing = True
                ret, frame = cap.read()
                if not ret:
                    self.processing = False
                    continue 
                self.frameid += 1
                frame_data = FrameData(self.camid, self.frameid, frame)
                detections=self.detector.detect(frame_data.vis_image)
                dets_to_sort = np.empty((0,6))
    
                for x1,y1,x2,y2,conf,detclass in detections:
                    dets_to_sort = np.vstack((dets_to_sort, 
                                np.array([x1, y1, x2, y2, conf, detclass])))
                # self.detector.draw_detections(frame_data.vis_image, detections=detections)
                tracked_dets = self.tracker.update(dets_to_sort)
                cropped_people = []
                features = []
                identities = []
                boxes = []
                if len(tracked_dets) > 0:
                    boxes = tracked_dets[:,:4]
                    identities = tracked_dets[:, 8]
                    for id, box in zip(identities, boxes):
                        pid = int(id)
                        x1, y1, x2, y2 = [int(x) if x >=0 else 0 for x in box]
                        person_image = frame_data.org_image[y1:y2, x1:x2]
                        feature = self.extractor.extract(person_image)[0]
                        features.append(feature)
                        cropped_people.append(person_image)
                        # save image and feature to db
                        saved_image_path = save_person_image(self.output_dir, self.camid, frame_data.frameid, pid, person_image, rotate_size=5)
                        leave_time = get_current_utc_iso()
                        if self.cam_db_search.is_existed(id=pid):
                            self.cam_db_search.update_old_id(pid, feature, leave_time, saved_image_path)
                        else:
                            self.cam_db_search.add_new_id(pid, feature,leave_time, leave_time, saved_image_path)
                    
                if self.visualize:
                    frame_data.cropped_people = cropped_people
                    frame_data.boxes = boxes
                    frame_data.identities = identities
                    frame_data.features = features
                    self.tracker.draw_track_dets(frame_data.vis_image, tracked_dets)
                    if self.framedata_manager is None:
                        break
                    self.framedata_manager.attach_framedata(frame_data)
                # logger.info(f'Camera {self.camid} is processing frame {self.frameid}')
                self.processing = False

            except Exception as err:
                logger.error(f"Camera [{self.camid}] has an error: {err}")  
                self.processing = False

            # except (KeyboardInterrupt, SystemExit):
            #     logger.error(f"Camera [{str(self.camid)}] is forced to stop by Ctr C")
            #     self.stop(by_SIGINT=True)

        logger.info(f"Camera [{str(self.camid)}] is stopped")


    def clear_old_output(self):
         # List all entries in the parent directory
        for item in os.listdir(self.output_dir):
            item_path = os.path.join(self.output_dir, item)
            # Check if the entry is a directory
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Delete the directory and its contents
        logger.info("Cleared all previous output images!!!")
    
    def search_feature(self, feature,start_time=None, end_time=None, num_candidates=3):
        resp = self.cam_db_search.search_feature(feature, num_candidates, start_time=start_time, end_time=end_time)
        print("====> Result in camera: ", self.camid)
        results = {}
        results['gids'] = []
        results['scores'] = []
        results['tracking_times'] = []
        images = []
        for i, res in enumerate(resp):
            # print(f'top {i+1}: ', ', id: ',res['_id'], ', score: ', res['_score'], ', ', 
            #       res['fields']['saved_image_path'][0],res['fields']['enter_time'][0], res['fields']['leave_time'][0] )  
            print(f'top {i+1}: ', ', id: ',res['_id'], ', score: ', res['_score'], ', ', 
                   res['_source'])    
            results['gids'].append(res['_id'])
            results['scores'].append(res['_score'])
            # results['tracking_times'].append({'enter_time': res['fields']['enter_time'][0],
            #                                   'leave_time': res['fields']['leave_time'][0]})
            save_image_path =  res['_source']['saved_image_path']
            image = read_image(save_image_path)
            if image is None:
                return None, None
            images.append(image)
        return results, images
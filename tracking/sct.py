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
from tracking.sort_tracker import SortTracker
from tracking.feature_extractor import FeatureExtractor

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
                # tracks = self.tracker.getTrackers()
                # self.tracker.draw_tracks(frame_data.vis_image, tracks)
                self.tracker.draw_track_dets(frame_data.vis_image, tracked_dets)
                if len(tracked_dets) > 0:
                    boxes = tracked_dets[:,:4]
                    identities = tracked_dets[:, 8]
                    features = []
                    cropped_people = []
                    for box in boxes:
                        x1, y1, x2, y2 = [int(x) if x >=0 else 0 for x in box]
                        person_image = frame_data.org_image[y1:y2, x1:x2]
                        feature = self.extractor.extract(person_image)
                        features.append(feature)
                        cropped_people.append(person_image)
                    frame_data.cropped_people = cropped_people
                    frame_data.boxes = boxes
                    frame_data.identities = identities
                    frame_data.features = features
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


        

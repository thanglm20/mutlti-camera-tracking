from threading import Thread, Lock, Event ,Condition
import time
import json
import numpy as np
import cv2
from utils.logger import Logger
logger = Logger()
from utils.config import Config
import sys
import socket
import zmq
from tracking.feature_extractor import FeatureExtractor

class ServerTask(Thread):
    def __init__(self, mct):
        self.mct = mct
        self.stopped = False
        self.stop_event = Event()
        self.processing = False
        Thread.__init__(self) 
        self.port = 5000
        self.context = None
        self.server = None
        self.cfg = Config.get_instance()
        self.results_map = {}

    def stop(self,  by_SIGINT=False):
        if self.server is not None:
            self.server.close()
        self.stopped = True
        self.stop_event.set()
        

    def run(self):
        """Main processing thead
        """
        self.context = zmq.Context()
        self.server = self.context.socket(zmq.REP)
        host = f"tcp://*:{ self.port}"
        self.server.bind(host)
        logger.info(f"Started server at: {host}")

        self.reid_gpu = self.cfg.reid.getboolean("gpu", True)
        self.reid_model = self.cfg.reid.getstr("reid_model", "./models/reid.onnx")
        device = 'cuda' if self.reid_gpu else 'cpu'
        self.extractor = FeatureExtractor(self.reid_model, device, 0)
        
        if self.stopped:
            return
        logger.info("Server is waiting ...")
        # Start looping
        while not self.stop_event.is_set() and not self.stopped:
            # try:    
                
                image = None
                image = self.recv_image()
                if image is not None:
                    feature = self.extractor.extract(image)[0]
                    results, images = self.mct.search(feature)
                    
                    if results is not None:
                        json_string = json.dumps(results)
                        print("searching result: ", json_string)
                        self.send_metadata(json_string, images)
                    else:
                        json_string = "None"
                        self.send_metadata(json_string, [np.zeros((1, 1))])
            # except Exception as err:
            #     logger.error(f"Server has an error: {err}")  
            #     self.processing = False
        logger.info("Stopped Server!!!")
    
    def recv_str(self):
        message = self.server.recv_string()
        return message
    
    def send_str(self, message):
        self.server.send_string(message)

    def send_image(self, image):
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()
        # Send the image
        self.server.send(image_bytes)

    def recv_image(self):
        image_bytes = self.server.recv()
        # Decode the byte array back into an image
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image

    def send_metadata(self, str, images):
        message_parts = []
        message_parts.append(str.encode('utf-8'))
        encoded_images = []
        for image in images:
            _, buffer = cv2.imencode('.jpg', image)
            encoded_images.append(buffer.tobytes())
        for image_bytes in encoded_images:
            message_parts.append(image_bytes)
        self.server.send_multipart(message_parts)

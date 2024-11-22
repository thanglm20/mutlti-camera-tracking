from threading import Thread, Lock, Event ,Condition
import time
import numpy as np
import cv2
from utils.logger import Logger
logger = Logger()
from utils.config import Config
import sys
import socket
import zmq

class ServerTask(Thread):
    def __init__(self):
        self.stopped = False
        self.stop_event = Event()
        self.processing = False
        Thread.__init__(self) 
        self.port = 5000
        self.context = None
        self.server = None

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
        self.server.bind("tcp://*:%s" % self.port)

        if self.stopped:
            return
        # Start looping
        while not self.stop_event.is_set() and not self.stopped:
            # try:    
                print("server is waiting ...")
                image = None
                image = self.recv_image()
                if image is not None:
                    cv2.imshow("Received Image", image)
                    cv2.waitKey(1)
                    # while not self.stop_event.is_set():
                    time.sleep(0.05)
                    # self.send_str(f'Received image with shape: {image.shape}')
                    # time.sleep(0.01)
                    # self.send_image(image)
                    json_string = '{"top1": 1, "top2": 2, "top3": 3}'
                    self.send_metadata(json_string, [image,image, image])
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


import cv2
import zmq
import threading
import time
from random import choice
import signal
import cv2
import os
import json
import numpy as np

class MctClient(object):
    def __init__(self) -> None:
        self.port = 5000
        self.init_client()

    def init_client(self):
        self.context = zmq.Context()
        print ("Connecting to server...")
        self.client = self.context.socket(zmq.REQ)
        self.client.connect ("tcp://localhost:%s" % self.port)
        self.client.setsockopt(zmq.RCVTIMEO, self.port)  # 5 seconds

    def send_str(self, message):
        self.client.send_string(message)
        #  Get the reply.
        # print("waiting from server ...")
        response = self.client.recv_string()
        return response
    
    def send_image(self, image):
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()
        # Send the image
        self.client.send(image_bytes)

    def recv_image(self):
        image_bytes = self.client.recv()
        # Decode the byte array back into an image
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    
    def wait_respone_str(self):
        response = self.client.recv_string()
        return response
    
    def recv_metadata(self):
        response = self.client.recv_multipart()
        str = response[0].decode('utf-8')
        data = json.loads(str)
        image_bytes_list = response[1:]
        images = []
        for image_bytes in image_bytes_list:
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            images.append(image)
        return data, images
def main():
    """main function"""
    client = MctClient()
    TEST_DIR = "./reid_data"
    


    list_people_images = []
    p = 'p1'
    list_file = os.listdir(os.path.join(TEST_DIR, p))
    list_people_images = [os.path.join(TEST_DIR, p, f) for f in list_file]
    for file_path in list_people_images:
        try:
            print ("Sending request ", file_path,"...")
            # message = client.send_str(f'Request {file_path} from client')
            # print ("Received reply ", "[", message, "]")
            img = cv2.imread(file_path)
            client.send_image(img)
            # response = client.wait_respone_str()
            
            data, images = client.recv_metadata()
            print(data)
            print(len(images))

        except Exception as e:
            raise
        except zmq.Again:
            print("Receive operation timed out!")
if __name__ == "__main__":
    main()

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
import random
import time
from datetime import datetime, timezone
import sys

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

    def close(self):
        self.client.close()

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
        if str == "None":
            return None, None
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
    
    image_path = sys.argv[1]
    print(f"Query image: {image_path}")
    # try:
        # list_people_images = []
        # p = 'p1'
        # list_file = os.listdir(os.path.join(TEST_DIR, p))
        # list_people_images = [os.path.join(TEST_DIR, p, f) for f in list_file]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    request_image = cv2.imread(image_path)
    client.send_image(request_image)
    # response = client.wait_respone_str()
    str, images = client.recv_metadata()
    if str:
        data = json.loads(str)
        print(data)
        print("Found a matching person with GID: ", data["gid"])
        desired_size = (210, 420) 
        output = []
        for camid, t, image in zip(data['cameras'], data['last_times'], images):
            utc_time_str = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
            print("\tAppears Camera: ", camid, " at time: ", \
                utc_time_str, " with image shape: ", image.shape)
            # Set a fixed height for all images
            resized_img = cv2.resize(image, desired_size)
            # Add text to the image
            cv2.putText(resized_img, f'Camera: {camid}', (10,30), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
            cv2.putText(resized_img, f'Last time: {utc_time_str}', (10,55), font, font_scale/2, (0, 255, 0), thickness, cv2.LINE_AA)
            output.append(resized_img)

        # Stack images horizontally
        horizontal_stack = np.hstack(output)
        height1, width1 = horizontal_stack.shape[:2]
        request_image = cv2.resize(request_image, desired_size)
        height2, width2 = request_image.shape[:2]
        if height1 != height2:
            request_image = cv2.resize(request_image, (width2, height1))
            height2, width2 = request_image.shape[:2]
        # Calculate padding for left and right
        total_padding = max(0, width1 - width2)
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding

        # Apply padding to request_image
        padded_request_image = cv2.copyMakeBorder(
            request_image,
            top=0,
            bottom=0,
            left=left_padding,
            right=right_padding,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)  # Black padding, adjust as needed
        )
        cv2.putText(padded_request_image, f'Global ID: {data["gid"]}', 
                    (int(padded_request_image.shape[1]/2-100),padded_request_image.shape[0]-20), 
                    font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
        cv2.putText(padded_request_image, f'Query image', (10,30), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
        # Vertically stack the images
        vertical_stack = np.vstack((padded_request_image, horizontal_stack))
        cv2.imshow("Matching Results", vertical_stack)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Not found")
        
        cv2.putText(request_image, f'Not Found', 
                    (int(request_image.shape[1]/2-100),request_image.shape[0]-20), 
                    font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
        cv2.putText(request_image, f'Query image', (10,30), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
        # Vertically stack the images
        cv2.imshow("Matching Results", request_image)
        cv2.waitKey(0)
    # except Exception as e:
    #     print("ERRor: ", e)
    #     client.close() 
    # for file_path in list_people_images:
    #     try:
    #         print ("Sending request ", file_path,"...")
    #         # message = client.send_str(f'Request {file_path} from client')
    #         # print ("Received reply ", "[", message, "]")
    #         img = cv2.imread(file_path)
    #         client.send_image(img)
    #         # response = client.wait_respone_str()
            
    #         data, images = client.recv_metadata()
    #         if()
    #         print(data)
    #         print(len(images))
    #         break
    #     except Exception as e:
    #         raise
    #     except zmq.Again:
    #         print("Receive operation timed out!")
if __name__ == "__main__":
    main()
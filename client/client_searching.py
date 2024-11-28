
import argparse
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

def padding_image(img, size):
    h, w = img.shape[:2]
    right_padding = size[0] - w
    bottom_padding = size[1] - h
    padded_request_image = cv2.copyMakeBorder(
        img,
        top=0,
        bottom=bottom_padding,
        left=0,
        right=right_padding,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # Black padding, adjust as needed
    )
    return padded_request_image

def add_left_lable(img, label):
    padded_request_image = cv2.copyMakeBorder(
        img,
        top=0,
        bottom=0,
        left=200,
        right=0,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # Black padding, adjust as needed
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    h, w = padded_request_image.shape[:2]
    cv2.putText(padded_request_image,label, (10, int(h/2)), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    return padded_request_image

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
        print(str)
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
    
    def send_metadata(self, str, images):
        message_parts = []
        message_parts.append(str.encode('utf-8'))
        encoded_images = []
        for image in images:
            _, buffer = cv2.imencode('.jpg', image)
            encoded_images.append(buffer.tobytes())
        for image_bytes in encoded_images:
            message_parts.append(image_bytes)
        self.client.send_multipart(message_parts)

def main(args):
    """main function"""
    client = MctClient()
    TEST_DIR = "./reid_data"
    
    image_path = args.query_image
    start_time = args.start_time
    end_time = args.end_time
    num_candidates = args.num_candidates
    request_data = {'start_time': start_time, 'end_time': end_time, 'num_candidates': num_candidates}
    request_data = json.dumps(request_data)
    print(f"Query image: {image_path}")
    try:
        # list_people_images = []
        # p = 'p1'
        # list_file = os.listdir(os.path.join(TEST_DIR, p))
        # list_people_images = [os.path.join(TEST_DIR, p, f) for f in list_file]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        request_image = cv2.imread(image_path)
        # client.send_image(request_image)
        client.send_metadata(request_data, [request_image])
        # response = client.wait_respone_str()
        results, result_images = client.recv_metadata()
        print(results)
        item_size = (150, 200)
        if results:
            idx_image = 0
            cam_result_images = []
            max_w = 0
            for camid, data in results.items():
                print(":--------------------------")
                gids = data['gids']
                scores = data['scores']
                num_images = len(gids)
                output_result = []
                cropped_images = result_images[idx_image: idx_image + num_images]
                for gid, img, score in zip(gids, cropped_images, scores):
                    resized_img = cv2.resize(img, item_size)
                    score = score * 100 
                    score = round(score, 2) if score <= 1.0 else 1.0
                    # Add text to the image
                    # cv2.putText(resized_img, f'Camera: {cam}', (10,30), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
                    cv2.putText(resized_img, f'score: {score}', (10,30), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
                    resized_img = padding_image(resized_img, (resized_img.shape[1] + 20,resized_img.shape[0]))
                    output_result.append(resized_img)
                idx_image = idx_image + num_images
                if len(output_result):
                    person_stack = np.hstack(output_result)
                    person_stack = add_left_lable(person_stack, camid)
                    max_w = max(max_w, person_stack.shape[1])
                    cam_result_images.append(person_stack)

            output_size = (max_w, item_size[1])
            output = []
            request_image = cv2.resize(request_image, item_size)
            request_image = add_left_lable(request_image, "Query image")
            request_image = padding_image(request_image, output_size)
            output.append(request_image)
            for i, img in enumerate(cam_result_images):
                img = padding_image(img, output_size)
                output.append(img)
            vertical_stack = np.vstack(output)
            for i, img in enumerate(cam_result_images):
                cv2.line(vertical_stack,(0, int((i+1)*output_size[1])),(output_size[0],int((i+1)*output_size[1])),(0,255,0),1)
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
    except Exception as e:
        print("ERRor: ", e)
        client.close() 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--query_image", default="", help="Query image path", type=str, required=True
    )
    parser.add_argument("--start_time", help="Start time searching",  type=str,default="")
    parser.add_argument("--end_time", help="End time searching", type=str, default="")
    parser.add_argument("--num_candidates", help="Top candidates", type=int, default=3)

    args = parser.parse_args()
    main(args)

'''
python .\client\client_searching.py --query_image=.\client\query_images\1.jpg --start_time=2024-11-28T06:22:59.232643+00:00 --end_time=2025-11-28T06:22:59.232643+00:00
'''
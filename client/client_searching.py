
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
def main():
    """main function"""
    client = MctClient()
    TEST_DIR = "./reid_data"
    
    image_path = sys.argv[1]
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
        client.send_image(request_image)
        # response = client.wait_respone_str()
        results, images = client.recv_metadata()
        item_size = (150, 200)
        if results:
            print("Found a matching person with GID: ", results["top_gid"])
            data = results["data"]
            idx = 0
            results_images= []
            max_w = 0
            for i, gid in enumerate(results["top_gid"]):
                print(":--------------------------")
                print(gid)
                output_result=[]
                num_images = len(data[f'{gid}']["cameras"])
                cameras = data[f'{gid}']["cameras"]
                last_times = data[f'{gid}']["last_times"]
                cropped_images = images[idx:idx+num_images]
                for cam, img, last_time in zip(cameras, cropped_images, last_times):
                    print(cam, img.shape, last_time)
                    utc_time_str = datetime.fromtimestamp(last_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
                    resized_img = cv2.resize(img, item_size)
                    # Add text to the image
                    cv2.putText(resized_img, f'Camera: {cam}', (10,30), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
                    cv2.putText(resized_img, f'{utc_time_str}', (10,55), font, font_scale/2, (0, 255, 0), thickness, cv2.LINE_AA)
                    resized_img = padding_image(resized_img, (resized_img.shape[1] + 20,resized_img.shape[0]))
                    output_result.append(resized_img)
                idx = idx + num_images
                person_stack = np.hstack(output_result)
                person_stack = add_left_lable(person_stack, f'Top{i + 1} ID: {gid}')
                max_w = max(max_w, person_stack.shape[1])
                results_images.append(person_stack)

            output_size = (max_w, item_size[1])
            output = []
            request_image = cv2.resize(request_image, item_size)
            request_image = add_left_lable(request_image, "Query image")
            request_image = padding_image(request_image, output_size)
            output.append(request_image)
            for i, img in enumerate(results_images):
                img = padding_image(img, output_size)
                output.append(img)
            vertical_stack = np.vstack(output)
            for i, img in enumerate(results_images):
                cv2.line(vertical_stack,(0, int((i+1)*output_size[1])),(output_size[0],int((i+1)*output_size[1])),(0,255,0),1)
            cv2.imshow("Matching Results", vertical_stack)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # desired_size = (210, 420) 
            # output = []
            # for camid, t, image in zip(data['cameras'], data['last_times'], images):
            #     utc_time_str = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
            #     print("\tAppears Camera: ", camid, " at time: ", \
            #         utc_time_str, " with image shape: ", image.shape)
            #     # Set a fixed height for all images
            #     resized_img = cv2.resize(image, desired_size)
            #     # Add text to the image
            #     cv2.putText(resized_img, f'Camera: {camid}', (10,30), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
            #     cv2.putText(resized_img, f'Last time: {utc_time_str}', (10,55), font, font_scale/2, (0, 255, 0), thickness, cv2.LINE_AA)
            #     output.append(resized_img)

            # # Stack images horizontally
            # horizontal_stack = np.hstack(output)
            # height1, width1 = horizontal_stack.shape[:2]
            # request_image = cv2.resize(request_image, desired_size)
            # height2, width2 = request_image.shape[:2]
            # if height1 != height2:
            #     request_image = cv2.resize(request_image, (width2, height1))
            #     height2, width2 = request_image.shape[:2]
            # # Calculate padding for left and right
            # total_padding = max(0, width1 - width2)
            # left_padding = total_padding // 2
            # right_padding = total_padding - left_padding

            # # Apply padding to request_image
            # padded_request_image = cv2.copyMakeBorder(
            #     request_image,
            #     top=0,
            #     bottom=0,
            #     left=left_padding,
            #     right=right_padding,
            #     borderType=cv2.BORDER_CONSTANT,
            #     value=(0, 0, 0)  # Black padding, adjust as needed
            # )
            # cv2.putText(padded_request_image, f'Global ID: {data["gid"]}', 
            #             (int(padded_request_image.shape[1]/2-100),padded_request_image.shape[0]-20), 
            #             font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
            # cv2.putText(padded_request_image, f'Query image', (10,30), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
            # # Vertically stack the images
            # vertical_stack = np.vstack((padded_request_image, horizontal_stack))
            # cv2.imshow("Matching Results", vertical_stack)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
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
    main()
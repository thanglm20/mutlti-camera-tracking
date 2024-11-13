

"""Frame data object

This module repesents for a data object, which is used to transfer between other
modules in the System system. The module uses the information stored in the object
to perform its functionality and save its output to the frame object.
"""
import time
import cv2
from queue import Queue

class FrameData():
    """Data object for each frame

    Each FrameData contains data for input frame, detection results and other status.

    Attributes:
        Frame information:
            frmidx, org_frame, time
        Pose result:
            org_poses, org_score
        Tracking result:
            track_data, track_score
        Action recognition result:
            action_list, action_class_id_list, result
    """
    def __init__(self, camid, frameid, image, poses=None, score=None):

        self.camid = camid
        self.frameid = frameid
        self.org_image = image.copy()
        self.vis_image = image
        self.time = time.time()

        self.identities = []
        self.boxes = []
        self.features = []
      
    def copy(self):
        return self  

    
    
class FramedataManager():
    def __init__(self, max_size=20):
        self.framedata_container = Queue(max_size)
        self.max_size = max_size
    
    def attach_framedata(self, frame:FrameData):
        if self.framedata_container.full() :
            self.framedata_container.get()
        self.framedata_container.put_nowait(frame)
        
    def pop_framedata(self) -> FrameData:
        framedata = None
        if not self.framedata_container.empty():
            framedata = self.framedata_container.get(block=False, timeout=1)
        return framedata
    
    def is_full(self):
        return self.framedata_container.full()
    
    def size(self):
        return self.framedata_container.qsize()
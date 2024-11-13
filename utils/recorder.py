import os
import cv2
from utils.logger import Logger
logger = Logger(loggername='recorder')
from utils.config import Config
from threading import Thread, Lock
import time
from queue import  Queue

class Recorder(object):
    def __init__(self, camid, video_name, num_queue=10, output_fps=10, output_shape=(640,360)):
        self.cfg = Config.get_instance()
        self.camcfg = self.cfg.get_section(camid)
        self.output_fps = output_fps
        self.output_shape = output_shape
        # To record output videos 
        self.record_frame_container = Queue(maxsize=num_queue)
        self.record_frame_container_original = Queue(maxsize=num_queue)
        self.video_name = video_name
        self.writer = None
        self.writer_original = None
        # Thread status
        self.is_running = True
        self.is_progress = False
        self.is_save_org = self.camcfg.getboolean('save_ori_video', False)
        self.is_save = self.camcfg.getboolean('save_video', False)

    def start(self):
        """Start Recorder Thread
        """
        self.thread = Thread(target=self.record_video, args=())
        self.thread.daemon = True
        self.mutex = Lock()
        self.thread.start()
        return self

    def put(self,frame, org_frame):
        """Send frame to Queue

        Args:
            frame (arr): Image array.
        """
        self.mutex.acquire()
        if self.is_save:
            if self.record_frame_container.full():
                d_frame = self.record_frame_container.get()
                del d_frame
                logger.error("Queue is full, skip one frame!")
            self.record_frame_container.put(frame)
        if self.is_save_org:
            if self.record_frame_container_original.full():
                d_frame = self.record_frame_container_original.get()
                del d_frame
                logger.error("Queue is full, skip one frame!")
            self.record_frame_container_original.put(org_frame)
        self.mutex.release()

    def record_video(self):
        """Record output video (full)

        Write the current frame to the video output on the drive. Two Supported formats are
        AVI and MP4 (select in configuration file).

        Args:
            image:
                Frame which is going to be written
            name_video:
                Name of output video
        """
        # Check whether recording is enable or not
        if self.is_save:
            self.writer = self.__init_writer("vis")
        if self.is_save_org:
            self.writer_original = self.__init_writer("org")
        # Init video writer
        if self.writer is not None or self.writer_original is not None:
            while self.is_running:
                try: 
                    self.is_progress = True
                    # Write frame to the output video
                    self.process()
                    self.is_progress  = False
                except (KeyboardInterrupt, SystemExit):
                    logger.error("[{}] Recorder is exiting...".format(str(self.camid)), exc_info=True)
                    self.release()
                    break
                
            
    def process(self):
        """Write frame to video. (Optional: Resized frame before writing)
        """
        time.sleep(0.005)
        if self.writer is not None:
            self.record_frame_container = self.write_frame(self.writer, self.record_frame_container)
        if self.writer_original is not None:
            self.record_frame_container_original = self.write_frame(self.writer_original, self.record_frame_container_original)

    def write_frame(self, writer, container):
        self.mutex.acquire()
        if not container.empty():
            frame = container.get()
            frame = cv2.resize(frame, self.output_shape)
            writer.write(frame)
        self.mutex.release()
        return container

    def __init_writer(self, ext="vis"):
        """Initial VideoWriter
        """
        logger.info('Recorder is running')
        # Choose video format: .avi or .mp4
        _format = self.camcfg.get('video_format', 'mp4')
        if _format == 'mp4':
            #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        # video name and where to store it
        video_dir = self.camcfg.get('video_dir', './videos')
        if not os.path.isdir(video_dir):
            os.makedirs(video_dir)
        file_name = f"{self.video_name}_{ext}.{_format}" #'{}.{}'.format(self.video_name + f"_{ext}", format)
        # init video writer
        writer = cv2.VideoWriter(os.path.join(video_dir, file_name), fourcc, self.output_fps, self.output_shape)
        logger.info("--> Writing videos to {} ...".format(os.path.join(video_dir, file_name)))
        return writer

    def release(self):
        """Release VideoWriter
        """
        self.is_running = False
        while not self.is_progress:
            time.sleep(0.01)
        # Free video writer to not damage the output video
        if self.writer:
            while not self.record_frame_container.empty():
                self.write_frame(self.writer, self.record_frame_container)
            logger.info("Recorder is being released.")
            time.sleep(2)
            self.writer.release()
            logger.info("Recorder is released.")
        if self.writer_original:
            while not self.record_frame_container_original.empty():
                self.write_frame(self.writer_original, self.record_frame_container_original)
            logger.info("Recorder original is being released.")
            time.sleep(2)
            self.writer_original.release()
            logger.info("Recorder original is released.")
        self.thread.join()
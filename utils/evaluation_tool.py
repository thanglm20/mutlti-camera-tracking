# """ THIS IS AN INDEPENDENT TOOL FOR RUNNING MULTIPLE VIDEO FOR:
# - Evaluation.
# - Get multiple output videos,logs.

# Run:
# $ cd ykk_workpose
# $ python3 -m utils.evaluation_tool
# """

# import time
# import sys
# import os
# import psutil
# import argparse
# import shutil
# from tqdm import tqdm
# from subprocess import Popen, PIPE

# sys.path.insert(0, './')
# from utils.config import Config
# from utils.logger import Logger
# logger = Logger(loggername='evaluation_tool')
# from utils.helper import *
# from libs.task.processor import Processor
# import math


# TRIAL = 0
# INPUT_FOLDER = 'data'
# OUTPUT_FOLDER = 'videos/'

# class EvaluationTool(object):
#     def __init__(self):
#         self.cfg = Config.get_instance()
#         self.encoding = "utf-8"
#         self.is_start = True
#         self.proc_list = {}

#         self.cam_list, self.multi_threads = self._get_cam_list()
#         self.list_process = {}
#         self.m_frame_container = dict()
#         for camid in self.cam_list:
#             self.m_frame_container[camid] = None

#         self.video_list = {}
#         self.video_idx = {}

#     def _get_cam_list(self):
#         cam_list = self.cfg.DEFAULT.get('cam_list', [])
#         print("Found camera_list: {}".format(cam_list))
        
#         multi_threads = 1
#         thread_num = self.cfg.evaluation.getint('thread_num', 1)
#         print("Multiple threads mode: {}".format(thread_num))
#         if thread_num > 1:
#             cam_list = ['cam01']
#             multi_threads = thread_num
#             for idx in range(2, thread_num+1):
#                 camid = 'cam'+str(idx).zfill(2)
#                 cam_list.append(camid)
#                 cam_path = 'configs/cam_configs/cam{}/{}.ini'.format(camid,camid)
#                 print('Making config file for {}...'.format(camid))
#                 shutil.copy('configs/cam_configs/cam01/cam01.ini', cam_path)
#                 lines = []
#                 with open(cam_path, 'r') as f:
#                     lines = f.readlines()
#                 with open(cam_path, 'w') as f:
#                     for l in lines:
#                         if l.startswith('['):
#                             f.write('[{}]\n'.format(camid))
#                         elif is_config_in_line('name', l):
#                             f.write('name = {}\n'.format(camid))
#                         else:
#                             f.write(l)
#         print("Final camera list: {}".format(cam_list))
#         print("Final multiple threads: {}".format(multi_threads))
#         return cam_list, multi_threads
    
#     def _run_pose_server(self):
#         """Run Pose Server"""
#         cuda_device = '0'
#         server_list = [[0]]
#         server_port = self.cfg.pose.getint('server_port', 50100)
        
#         for idx, servers in enumerate(server_list):
#             device_id = cuda_device[idx]
#             for server_id in servers:
#                 port = server_port + server_id
#                 try:
#                     is_running = False
#                     for p in psutil.process_iter():
#                         if p.cmdline() == ['python3', '-m', 'libs.pose.pose_server',
#                                                 '--port', '{}'.format(port)]:
#                             is_running = True
#                             break
#                     if is_running:
#                         continue 

#                     logger.info("Starting pose_server #{} ...".format(server_id))
#                     pose_server = Popen(['python3', '-m', 'libs.pose.pose_server',
#                                         '--port', '{}'.format(port)], 
#                                         stdin=PIPE, env=dict(CUDA_VISIBLE_DEVICES=str(device_id), **os.environ), encoding=self.encoding)
#                     self.proc_list[pose_server.pid] = pose_server
#                     logger.info("Started pose_server #{} successfully!".format(server_id))
#                 except:
#                     logger.error("Error when starting pose_server #{}!".format(server_id))

#     def _run_camera(self, camid):
#         self.m_frame_container[camid] = None
#         self.list_process[camid] = Processor(camid, self.m_frame_container)
#         self.list_process[camid].start()

#     def _get_video_list(self, camid):
#         camcfg = self.cfg.get_section(camid)
#         input_folder = camcfg.get('input_folder', INPUT_FOLDER)
#         self.video_list[camid] = []
#         self.video_idx[camid] = 0
#         try:
#             for name in os.listdir(input_folder):
#                 if name.endswith(".mp4") or name.endswith(".mkv") or name.endswith(".avi"):
#                     name = name.rstrip().lstrip()
#                     path = os.path.join(input_folder, name)
#                     if os.path.isfile(path):
#                         self.video_list[camid].append(name)
#                     else:
#                         logger.warning('[{}] Video {} is not exists!'.format(camid, path))
#         except Exception as e:
#             print(e)
#         self.video_list[camid] = sorted(self.video_list[camid])
#         # self.video_list[camid] = [
#         #     '20221007_184745_CE1D.mkv',
#         #     '20221007_180732_EA60.mkv',
#         #     '20221007_200813_87D3.mkv',
#         #     '20221007_210332_667C.mkv',
#         #     '20221007_195810_E58E.mkv'
#         # ]
#         if len(self.video_list[camid]):
#             logger.info("[{}] Running evaluation for {} videos ...".format(camid, len(self.video_list[camid])))
#         else:
#             logger.warning("[{}] No video for running evaluation".format(camid))

#     def _set_next_video(self, camid):
#         camcfg = self.cfg.get_section(camid)
#         input_folder = camcfg.get('input_folder', INPUT_FOLDER)
#         output_folder = camcfg.get('output_folder', OUTPUT_FOLDER)
#         index = self.video_idx[camid]
#         total = len(self.video_list[camid])
#         if index < total:
#             print("=========================================================================================")
#             logger.info("[{}] -> Running video {} ({}/{})...".format(camid, self.video_list[camid][index], (index+1), total))
#             print("=========================================================================================")
#             video_url = self.video_list[camid][index]
#             video_name = os.path.basename(video_url)
#             video_folder = os.path.dirname(video_url)
#             video_name = os.path.splitext(video_name)[0]
#             video_folder = os.path.join(output_folder, video_folder)
#             self.video_idx[camid] += 1

#             lines = []
#             with open('./configs/cam_configs/{}/{}.ini'.format(camid,camid), 'r') as f:
#                 lines = f.readlines()
#             with open('./configs/cam_configs/{}/{}.ini'.format(camid,camid), 'w') as f:
#                 for l in lines:
#                     if is_config_in_line('camera_url', l):
#                         f.write('camera_url = {}\n'.format(os.path.join(input_folder, video_url)))
#                     elif is_config_in_line('save_video', l):
#                         f.write('save_video = True\n')
#                     else:
#                         f.write(l)

#             Config.get_instance().reload()
#             return True
#         else:
#             logger.info("[{}] No more videos to run!".format(camid))
#         return False

#     def _init_configs(self):
#         lines = []
#         with open('./configs/system_config/config.ini', 'r') as f:
#             lines = f.readlines()
#         with open('./configs/system_config/config.ini', 'w') as f:
#             for l in lines:
#                 if is_config_in_line('camera_list', l):
#                     if self.multi_threads == 1:
#                         f.write(l)
#                     else:
#                         cam_ids = [int(camid.split('cam')[1]) for camid in self.cam_list]
#                         f.write('camera_list = {}-{}\n'.format(cam_ids[0], cam_ids[-1]))
#                 elif is_config_in_line('eval_enable', l):
#                     f.write('eval_enable = True\n')
#                 else:
#                     f.write(l)

#         lines = []
#         with open('./configs/cam_configs/COMMON.ini', 'r', encoding='utf8') as f:
#             lines = f.readlines()
#         with open('./configs/cam_configs/COMMON.ini', 'w', encoding='utf8') as f:
#             for l in lines:
#                 if is_config_in_line('show_fps', l):
#                     f.write('show_fps = True\n')
#                 elif is_config_in_line('video_format', l):
#                     f.write('video_format = mp4\n')
#                 else:
#                     f.write(l)
        
#         Config.get_instance().reload()

#     def _restore_configs(self):
#         lines = []
#         with open('./configs/system_config/config.ini', 'r') as f:
#             lines = f.readlines()
#         with open('./configs/system_config/config.ini', 'w') as f:
#             for l in lines:
#                 if is_config_in_line('eval_enable', l):
#                     f.write('eval_enable = False\n')
#                 else:
#                     f.write(l)

#         lines = []
#         with open('./configs/cam_configs/COMMON.ini', 'r', encoding='utf8') as f:
#             lines = f.readlines()
#         with open('./configs/cam_configs/COMMON.ini', 'w', encoding='utf8') as f:
#             for l in lines:
#                 if is_config_in_line('video_format', l):
#                     f.write('video_format = mp4\n')
#                 else:
#                     f.write(l)

#         for camid in self.list_process:
#             lines = []
#             with open('./configs/cam_configs/{}/{}.ini'.format(camid,camid), 'r') as f:
#                 lines = f.readlines()
#             with open('./configs/cam_configs/{}/{}.ini'.format(camid,camid), 'w') as f:
#                 for l in lines:
#                     if is_config_in_line('save_video', l):
#                         f.write('save_video = True\n')
#                     else:
#                         f.write(l)

#         Config.get_instance().reload()

#     def _start(self):
#         logger.info("Starting evaluation...")
#         pbar = tqdm(total=0, position=0, bar_format='{desc}')

#         self._init_configs()
#         self._run_pose_server()
        
#         running_cams = 0
#         for camid in self.cam_list:
#             self._get_video_list(camid)
#             print("Found {} videos for {}".format(len(self.video_list[camid]), camid))

#         if self.multi_threads > 1:
#             all_video_list = self.video_list['cam01']
#             cam_per_thread = int(math.ceil(len(all_video_list) / self.multi_threads))
#             for idx, camid in enumerate(self.cam_list):
#                 start_idx = idx * cam_per_thread
#                 end_idx = min((idx+1) * cam_per_thread, len(all_video_list))
#                 self.video_list[camid] = all_video_list[start_idx:end_idx]
#                 print("=> New videos list for {}: {}-{}, {}".format(camid, start_idx, end_idx, self.video_list[camid]))
            
#             Config.get_instance().reload()

#         for camid in self.cam_list:
#             has_video = self._set_next_video(camid)
#             if has_video:
#                 self._run_camera(camid)
#                 running_cams += 1
#         if running_cams == 0:
#             logger.warning("No camera available for running evaluation!")
#             self._stop()
        
#         start_time, cnt = time.time(), 0
#         while self.is_start:
#             start = time.time()
#             try:
#                 running_cams = 0
#                 for camid in self.list_process:
#                     if not self.list_process[camid].is_running:
#                         has_video = self._set_next_video(camid)
#                         if has_video:
#                             self._run_camera(camid)
#                             running_cams += 1
#                     else:
#                         running_cams += 1

#                 if running_cams == 0:
#                     self._stop()

#                 #4. Show fps
#                 cnt += 1
#                 if cnt == 10:
#                     strout = 'Processing (FPS)'
#                     for camid in self.list_process:
#                         strout += ' [{}: {:0.2f}]'.format(camid, self.list_process[camid].fps)
#                     pbar.set_description(strout)
#                     cnt = 0

#                 time.sleep(max(0, 3-(time.time()-start)))
#             except (KeyboardInterrupt, SystemExit):
#                 print("Exiting...")
#                 self._stop()
#                 break

#     def _stop(self):
#         logger.info("Done evaluation!")
#         try:
#             for camid in self.list_process:
#                 logger.info('Stopping camera {}...'.format(camid))
#                 self.list_process[camid].stop()

#             for _, process in self.proc_list.items():
#                 cmd = ' '.join(arg for arg in process.args)
#                 logger.info('Stopping process "{}"...'.format(cmd))
#                 process.kill()
        
#         except Exception as e:
#             logger.warning('ERROR on Stop Main: ', e)
        
#         self._restore_configs()
#         exit()

# def is_config_in_line(config_name, line):
#     try:
#         return line.startswith(config_name) and '=' in line and line.split('=')[0].strip()==config_name
#     except:
#         return False

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()

#     logger.info("Running Evaluation tool...")
#     evaluator = EvaluationTool()
#     evaluator._start()

""" Define common variables and functions which are used in whole projects."""

from datetime import datetime, timedelta
import time
import os
import numpy as np 

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (88,88,88)

# A list of multiple pairs of body parts.
L_PAIR = [
            [0, 0, "NO", "NOSE"], [0, 1, "RY", "Right Eye"], [0, 2, "LY", "Left Eye"], [1, 3, "RE", "Right Ear"],
            [2, 4, "LE", "Left Ear"], [3, 5, "RS", "Right Shoulder"], [4, 6, "LS", "Left Shoulder"],
            [5, 7, "REl", "Right Elbow"], [6, 8, "LEl", "Left Elbow"], [7, 9, "RW", "Right Wrist"],
            [8, 10, "LW", "Left Wrist"], [5, 11, "RH", "Right Hip"], [6, 12, "LH", "Left Hip"],
            [11, 13, "RK", "Right Knee"], [12, 14, "LK", "Left Knee"], [13, 15, "RA", "Right Ankle"],
            [14, 16, "LA", "Left Ankle"], [5, 6, "LS", "Left Shoulder"], [11, 12, "LH", "Left Hip"]
        ]
        
# A list of visualization colors for each body part.
LINE_COLOR = [(255, 0, 127),
              (255, 0, 127),
              (255, 0, 127),
              (0, 204, 255),
              (0, 204, 255),
              (0, 204, 255),
              (229, 79, 255),
              (229, 79, 255),
              (229, 79, 255),
              (0, 255, 0),
              (0, 255, 0),
              (0, 255, 0),
              (255, 0, 0),
              (255, 0, 0),
              (255, 0, 0),
              (0, 0, 255),
              (0, 0, 255),
              (0, 0, 255),
              (0, 0, 255),
              (0, 0, 255),
              (0, 0, 255)]

class Singleton(object):
    """
    Singleton interface:
    http://www.python.org/download/releases/2.2.3/descrintro/#__new__
    
    ----
    Using for UniqueConfigParser (utils.configs_utils) and LoggerManager (utils.helper)
    """
    def __new__(cls, *args, **kwds):
        it = cls.__dict__.get("__it__")
        if it is not None:
            return it
        cls.__it__ = it = object.__new__(cls)
        it.init(*args, **kwds)
        # print('Initialize %s: [id: %d][args: %s] [kwargs: %s]'%(repr(cls), id(it), str(args), str(kwds)))
        return it

    def init(self, *args, **kwds):
        pass

class imdict(dict):
    def __hash__(self):
        return id(self)

    def _immutable(self, *args, **kws):
        raise TypeError('object is immutable')
    
    __setitem__ = _immutable
    __delitem__ = _immutable
    clear       = _immutable
    update      = _immutable
    setdefault  = _immutable
    pop         = _immutable
    popitem     = _immutable

def write_csv(filepath, mess, header):
  """Write csv"""
  dirname = os.path.dirname(filepath)
  os.makedirs(dirname,  exist_ok=True)
  if not os.path.exists(filepath):
    with open(filepath, 'w+') as f:
      f.write(','.join(header))
      f.write('\n')
  with open(filepath, 'a') as f:
      f.write(','.join(mess))
      f.write('\n')

def write_action_log(name, camid, save_path = './logs/evaluation', _workspace='', _frame_id='', _action=''):
  """Write action log"""
  header = ['FRAME_ID', 'ACTION']
  mess = [str(_frame_id), str(_action)]
  
  file_csv_path = os.path.join(save_path,'{}@{}@{}.csv'.format(name, camid, _workspace))
  write_csv(file_csv_path, mess, header)

def write_summary_action_log(name, camid, save_path = './logs/summary', _workspace='', _frame_start='',_frame_end='',
                              _start_time=None, _end_time=None,
                              _action='', _action_time=''):
  """Write action with summary time log"""
  header = ['FRAME_START', 'FRAME_END', 'TIMESTAMP_START', 'TIMESTAMP_END,' 'ACTION', 'ACTION_TIME']
  mess = [str(_frame_start), str(_frame_end),timestamp2datestring(_start_time), timestamp2datestring(_end_time), str(_action), str(_action_time)]
  
  file_csv_path = os.path.join(save_path,'{}@{}@{}.csv'.format(name, camid, _workspace))
  write_csv(file_csv_path, mess, header)

def check_csv_file(name, camid, save_path='./logs/evaluations', _workspace=''):
  """Check if csv file is existed"""
  file_csv_path = os.path.join(save_path,'{}@{}@{}.csv'.format(name, camid, _workspace))
  if os.path.exists(file_csv_path):
        os.remove(file_csv_path)

def timestamp2datestring(timestamp):
    '''
    Convert timestamp to string with format YYYY-mm-dd HH:MM:SS
    '''
    str_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp)) + '.' + str(round(timestamp, 2)).split('.')[-1]
    return str_date

def datestring2timestamp(str_date):
    '''
    Convert from string with format YYYY-mm-dd HH:MM:SS to timestamp format
    '''
    timestamp = time.mktime(time.strptime(str_date.split('.')[0], '%Y-%m-%d %H:%M:%S')) + float(str_date.split('.')[-1][:2])/100
    return timestamp

def is_grpc_ready(channel):
    """This function checks if grpc is running or not.

    Args:
      channel: A channel provides a connection to a gRPC server on a specified host and port.

    Returns:
      A boolean indicating if grpc is running or not.

    Raises:
      IOError: An error occurred when exceeding deadline but haven't got any result.
    """

    import grpc
    try:
        grpc.channel_ready_future(channel).result(timeout=60)
        return True
    except grpc.FutureTimeoutError:
        return False
  
def filter_asilla_pose(poses, scores, pose_score_threshold=0.0005, sum_scores_threshold=2.0):
    """
    Filter asilla poses

    Args:
        poses (array): poses 
        scores (array): scores of poses
        pose_score_threshold (float, optional): Minimum accept filter scores. Defaults to 0.001.

    Returns:
        new_poses, new_scores: filter poses and score
    """
    remove_indexs = []
    number_pose, number_pose_keypoint = scores.shape
    # print("scores:", scores.sum(axis=1))
    
    if poses.shape[1] == 17:
        importain_point = [0,5,6,7,8,9,10,11,12]
    else:
        importain_point = [0,1,2,3,4,5,6,7,8,11]
        
    for i in range(number_pose):
        # remove score pose with threshold
        number_bad_score =  np.count_nonzero(scores[i][importain_point] < pose_score_threshold)
        if number_bad_score != 0:
            # print("number_bad_score")
            remove_indexs.append(i)
            continue
        # remove to many miss keypoint
        if np.count_nonzero(poses[i]) < number_pose_keypoint:
            # print("count_nonzero")
            remove_indexs.append(i)
            continue 
        
        # remove if score of pose is lowersum_scores_threshold
        # print(scores[i][importain_point].sum())
        if scores[i][importain_point].sum() <= sum_scores_threshold:
            remove_indexs.append(i)
            continue       
    #print("remove pose_indexs:", remove_indexs)
    new_poses = np.delete(poses, remove_indexs, axis=0)
    new_scores = np.delete(scores, remove_indexs, axis=0)
    
    return new_poses, new_scores

def get_video_name_from_camera_url(camera_url):
    """ Get camera_url and handle 
    step1: check if file, path(usb) or rtsp(ipcam)
    step2: check if add time suffix
    
    Arguments:
      camera_url (str): get from camcfg: camera_url
        eg:
          'data/video.mkv'
          '/dev/video0'
          'rtsp://192.168.x.x:554/1/'
    Returns:
      video_name: (str)video name without extension
      video_name_with_time: (str) video name without extension and with time suffix
    """
    video_name_with_ext = camera_url.split('/')
    if os.path.isfile(camera_url):
        video_name = os.path.splitext(video_name_with_ext[-1])[0]
    elif os.path.exists(camera_url) or 'rtsp' in camera_url:
        video_name = '_'.join([part for part in video_name_with_ext if part])
    else:
        video_name = os.path.splitext(video_name_with_ext[-1])[0]
    video_name_with_time = f'{video_name}@{time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))}'
    
    return video_name, video_name_with_time

"""All functions to process and get parameters from config files."""

import os
import time
import glob
import numpy as np
from threading import Thread, Lock
from utils.configs_utils import UniqueConfigParser


class Config(object):
    """This class provides multiple functions to get and process data from config files.

    Attributes:
        ini_file: A string indicating path to config file. 
        cam_folder: A string indicating path to camera config folder. 
        all_ini_file: A string indicating path to all_configs file which contains data of all config files.
        cam_list: A list of all cameras.
        reload_files: A list of reloadable files.
        config_times: A list containing last modified timestamps of config files.
        reload_time: A period of time to check modified config files.
        auto_reload: A boolean indicating if checking reloadable files or not. 
    """

    _instance = None

    @staticmethod
    def get_instance(ini_file="./configs/config.ini", parent_path=os.getcwd()):
        filepath = os.path.join(parent_path, ini_file)
        print("Reading config path: ",filepath)
        
        if Config._instance is None:
            config = Config(filepath=filepath, parent_path=parent_path)
            Config._instance = config
        return Config._instance

    def __init__(self, filepath="./configs/config.ini", auto_reload = True, parent_path=os.getcwd()):
        super(Config, self).__init__()
        self.ini_file = filepath
        # # self.cam_folder = os.path.join(parent_path, './configs/cameras')
        # # self.common_file = os.path.join(parent_path, './configs/COMMON.ini')
        self.all_ini_file = os.path.join(parent_path, './configs/.all_configs.ini')
        # self.cam_list = []

        self._write_all_configs()
        # self.reload_files = [self.ini_file] + glob.glob('{}/*.ini'.format(self.cam_folder))
        self.reload_files = [self.ini_file] 
        self._configs = UniqueConfigParser(filepath=self.all_ini_file)
        # self._parse_cam_configs()

        self.config_times = []
        for cfg_file in self.reload_files:
            self.config_times.append(os.path.getmtime(cfg_file))

        self.reload_time = 10
        self.auto_reload = auto_reload
        self.thread = Thread(target=self._auto_reload, args=())
        self.thread.daemon = True
        self.mutex = Lock()
        self.thread.start()

    def _write_all_configs(self):
        """Write data from all config files to a single file."""

        # cam_files = glob.glob('{}/*.ini'.format(self.cam_folder))
        # all_config_files = [self.ini_file] + cam_files + [self.common_file]
        all_config_files = [self.ini_file]

        with open(self.all_ini_file, "wb") as all_configs_file:
            for f in all_config_files:
                with open(f, "rb") as cfg_file:
                    all_configs_file.write(cfg_file.read())
                all_configs_file.write(b'\n')

    def _parse_cam_configs(self):
        # Get list of camera sections [camxx]
        cam_keys = []
        for k in self._configs._sub_configs.keys():
            if 'cam' in k:
                cam_keys.append(k)
        cam_keys.sort()

        # Add cam_list to DEFAULT section
        default = {}
        for k,v in self._configs.DEFAULT.items():
            default[k] = v

        cam_list = self.range_to_list(self._configs.input.get('camera_list', ''))
        if not cam_list:
            self.cam_list = cam_keys
        else:
            self.cam_list = []
            for cam_key in cam_keys:
                try:
                    cam_idx = int(cam_key.split('cam')[1])
                    if cam_idx in cam_list:
                        self.cam_list.append(cam_key)
                except: pass
        default['cam_list'] = self.cam_list
        self._configs.DEFAULT._update(default)

        # Update common configs for each camera:
        for camid in self.cam_list:
            cam_cfgs = {}
            for k,v in self._configs[camid].items():
                cam_cfgs[k] = v
            for k,v in self._configs.COMMON.items():
                if not k in cam_cfgs.keys():
                    cam_cfgs[k] = v
            
            self._configs[camid]._update(cam_cfgs)

        # Remove temp .ini file
        try:
            os.remove(os.path.abspath(self.all_ini_file))
        except:
            pass

    def _auto_reload(self):
        """Reload configs after an amount of time"""
        while self.auto_reload:
            self.reload()
            time.sleep(self.reload_time)

    def load_manual(self):
        """Load configs manually."""
        self._configs.reset(filepath=self.ini_file)
        print(self.to_string())

    def reload(self):
        """Check if config files are modified or not and then reload modified files."""

        for idx, cfg_file in enumerate(self.reload_files):
            current_time = os.path.getmtime(cfg_file)
            if current_time != self.config_times[idx]:
                print('Detected changes in: {}. Reloading configs...'.format(cfg_file))
                while True:
                    self._write_all_configs()
                    if os.path.isfile(self.all_ini_file):
                        try:
                            self._configs.reset(filepath=self.all_ini_file)
                            break
                        except Exception as e:
                            print('Error on reset configs:\n', e)
                    time.sleep(1)
                # self._parse_cam_configs()
                for idx2, cfg_file2 in enumerate(self.reload_files):
                    self.config_times[idx2] = os.path.getmtime(cfg_file2)
                break
    
    def __getitem__(self, index):
        """Get item from configs"""
        return self._configs[index]
    
    def __getattr__(self, attribute):
        """Get attribute from configs"""
        try:
            return self.__getattribute__(attribute)
        except:
            return self._configs[attribute]

    def to_dict(self):
        """Get sub_configs"""
        return self._configs._sub_configs

    def get_section(self, sect_name = "DEFAULT"):
        """ Return an object containing data in DEFAULT section from the config file."""
        return self[sect_name]

    def get_abr_int(self, opt_name, default=None):
        """ Return a value from abr section in config file for given parameters."""
        return self.get_section("abr").getint(opt_name, default)

    def get_default_int(self, opt_name, default=None):
        """ Return an integer from the config file for given parameters."""
        return self.get_section("DEFAULT").getint(opt_name, default)

    def get_default_bool(self, opt_name, default=None):
        """ Return a boolean from the config file for given parameters."""
        return self.get_section("DEFAULT").getboolean(opt_name, default)

    def get_default_float(self, opt_name, default=None):
        """ Return a float from the config file for given parameters."""
        return self.get_section("DEFAULT").getfloat(opt_name, default)

    def get_default(self, opt_name, default=""):
        """ Return a string from the config file for given parameters."""
        return self.get_section("DEFAULT").getstr(opt_name, default)
    
    def to_string(self):
        """ Return string of object """
        return repr(self)

    def range_to_list(self, data):
        """ Convert a list of camera ranges to a camera list.
        
        Args:
            data: Multiple ranges of cameras.
            example:

            [1-5,7,9,12-15]

        Returns:
            A list of valid cameras.
            example:

            [1,2,3,4,5,7,9,12,13,14,15] 

        Raises:
            IOError: An error occurred appending camid in camera list or converting a list of ranges to a list.
        """

        if ',' in data:
            tmp = data.split(',')
        else:
            tmp = [data]
        
        result = []
        for idx in tmp:
            idx = idx.lstrip().rstrip()
            if not '-' in idx:
                try:
                    result.append(int(idx))
                except: pass
            else:
                A,B = idx.split('-')
                try:
                    A = int(A)
                    B = int(B)
                    result += list(range(A, B+1))
                except: pass
        
        return result

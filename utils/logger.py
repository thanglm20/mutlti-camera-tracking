from __future__ import print_function
import os
import logging
import sys
from logging.handlers import RotatingFileHandler
import datetime, time
import json
from utils.helper import Singleton
from utils.config import Config


class LoggerManager(Singleton):
    """
    Logger Manager.
    Handles all logging files.
    """
    def init(self, loggername='root', stdout=False, **kwargs):
        self.kwargs = kwargs
        self.loggername = loggername
        self.logger = logging.getLogger(loggername)
        rhandler = None
        
        log_path = kwargs.get('log_path', os.getcwd())
        # log_name = kwargs.get('log_name', '%s.log.%s'%(loggername,datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
        log_name = kwargs.get('log_name', f'{loggername}.log')
        
        log_dir = kwargs.get('log_dir', 'logs')
        logtofile = kwargs.get('logtofile', True)
        os.makedirs(os.path.join(log_path, log_dir), exist_ok=True)
        LOGFILE = os.path.join(log_path, os.path.join(log_dir,log_name))
        formatter = kwargs.get('formatter', None)
        level = kwargs.get('level', "info")
        
        if formatter is None:
           formatter = logging.Formatter(
               fmt = '[%(asctime)s][%(filename)s:%(lineno)d][%(funcName)s][%(threadName)s][%(levelname)s]::%(message)s',
               datefmt = '%F %H:%M:%S'
           )

        max_byte = kwargs.get('max_byte', 50 * 1024 * 1024)
        backup_cnt = kwargs.get('backup_count', 5)
        
        if logtofile:
            try:
                rhandler = RotatingFileHandler(
                    LOGFILE,
                    mode='a',
                    maxBytes = max_byte,
                    backupCount=backup_cnt
                )
            except:
                print(IOError("Couldn't create/open file \"" + \
                          LOGFILE + "\". Check permissions."))
                rhandler = logging.StreamHandler(sys.stdout)

        if stdout:
               stdout_handler = logging.StreamHandler(sys.stdout)
               stdout_handler.setFormatter(formatter)
               self.logger.addHandler(stdout_handler)

        logging.addLevelName(logging.INFO, "\033[32m%s\033[0m" % logging.getLevelName(logging.INFO))
        logging.addLevelName(logging.DEBUG, "\033[36m%s\033[0m" % logging.getLevelName(logging.DEBUG))
        logging.addLevelName( logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
        logging.addLevelName( logging.CRITICAL, "\033[7;31;31m%s\033[0m" % logging.getLevelName(logging.CRITICAL))
        logging.addLevelName( logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))

        self.logger.setLevel(logging.INFO)
        if level=='debug':
            self.logger.setLevel(logging.DEBUG)
        elif level=='error':
            self.logger.setLevel(logging.ERROR)
        elif level=='critical':
            self.logger.setLevel(logging.CRITICAL)
        elif level=='warning':
            self.logger.setLevel(logging.WARNING)


        if logtofile:
            rhandler.setFormatter(formatter)
            self.logger.addHandler(rhandler)
        self.logger.info('Start logging into : %s'%LOGFILE)
        self.logger.propagate = False

    def debug(self, loggername, msg):
        #self.logger = logging.getLogger(loggername)
        self.logger.debug(msg)

    def error(self, loggername, msg):
        #self.logger = logging.getLogger(loggername)
        self.logger.error(msg)

    def info(self, loggername, msg):
        #self.logger = logging.getLogger(loggername)
        self.logger.info(msg)

    def warning(self, loggername, msg):
        #self.logger = logging.getLogger(loggername)
        self.logger.warning(msg)

        
class LoggerABC(object):
    """
    Singleton Logger Class:
    init: Initialize object
      args:
            loggername[string]: Name the the logger
            stdout[boolean]: print out the standard output or terminal
      kwargs:
            log_path[string]: path of the log directory
                default: working path (os.getcwd())
            log_dir[string]: directory of the log file
                default: logs
            log_name[string]: file name of the log file
                default: 'loggername_YYYYmmmdd_HH.log'
            formatter[string]: to set format of the logger instead of default
                default: [%(asctime)s][%(filename)s:%(lineno)d][%(funcName)s][%(threadName)s][%(levelname)s]::%(message)s'
            level[string]: to set these level debug, error, warning, critical
                default: info
    re_init: Reinitialize with the new arguments
        params the same as init func
    
    Note:
        Must initialize at the beginning of the main of the program
        After that, the next init will be referenced to the first one
    """
    def __init__(self, loggername="root", stdout=False, **kwargs):
        self.lm = LoggerManager(loggername, stdout, **kwargs) # LoggerManager instance
        self.loggername = loggername # logger name
        self.kwargs = kwargs
        # self.debug('Link logger "%s" into "%s"'%(self.loggername, self.lm.loggername))
    
    def reinit(self, loggername='root', stdout=True, **kwargs):
        for h in self.logger.handlers:
            self.logger.removeHandler(h)
        LoggerManager.__it__ = None
        self.lm = LoggerManager(loggername, stdout, **kwargs)

    @property
    def logger(self):
        return self.lm.logger
    
    @property
    def debug(self):
       return self.logger.debug

    @property
    def info(self):
        return self.logger.info

    @property
    def error(self):
       return self.logger.error

    @property
    def warning(self):
       return self.logger.warning
    
    @property
    def critical(self):
       return self.logger.critical
      
class TimeInspector(Singleton):
    '''
    TimeInspector Decorator Class
    init: Initialize object
     args:
         threshold: threshold of average time consumption to show
             default: 0.
         skip: If true, skip inspecting time consumption to save resources
             default: True
    set_threshold:
     args:
         threshold: new threshold value
    set_skip: skip inspecting time consumption
    Usage:
           # use as an decorator
           caltime = TimeInspector()
           func = lambda x: x
           func = caltime(func)
           use: print(caltime) # json format
           key words:
                - count: number time the func has been called
                - total_time: total consumption time by accumulating all times
                - avg_time: total_time/count
                - min: min consumption time
                - max: max consumption time

    '''
    __skip = False
    __single = True
    def __init__(self, threshold=0., skip=True):
        if TimeInspector.__single:
           self.data = {}
           if skip:
              self.set_skip()
           self.threshold = threshold
           self.count = 0 
           self.total_time = 0
           TimeInspector.__single = False
    
    def set_threshold(self, threshold):
        """Set the threshold

        Args:
            threshold (float): Threshold
        """
        self.threshold = threshold

    def set_skip(self):
        """Skip/Ignore cal time
        """
        if TimeInspector.__skip == False:
            TimeInspector.__skip = True
        
    def _get_name(self, func):
        """Get name of the function.

        Args:
            func (method): Function.

        Returns:
            str: Function name.
        """
        string = repr(func).split(" ")[1]
        return string
    
    
    def __repr__(self):
        """Return string of cal time data

        Returns:
            str: Cal time data.
        """
        data = {}
        self.count = 1 if self.count == 0 else self.count
        self.data['caltime'] = {'count': self.count, 'total_time': self.total_time, 'avg_time': self.total_time/self.count}
        for item in self.data.items():
            key, val = item
            if val['avg_time'] > self.threshold:
               data[key] = val
        data = json.dumps(data, indent=2)
            
        return '<%s.%s object at %s>\n%s' % (
        self.__class__.__module__,
        self.__class__.__name__,
        hex(id(self)),
        data)
    
    def _init_dict(self):
        """Initial Cal time

        Returns:
            dict: Cal time.
        """
        return {'count':0, 'total_time': 0, 'avg_time': 0, 'min':99999 , 'max':0}
    
    def __call__(self, func):
        """Calculate time running the func

        Args:
            func (method): Func which time is calculated.
        """
        def decorator(*args, **kwargs):
            if not TimeInspector.__skip:
               start_count = time.time() 
               func_name = self._get_name(func)
               data_dict = self.data.get(func_name, self._init_dict())
               start = time.time()
               result = func(*args, **kwargs)
               duration = time.time() - start
               data_dict['count'] += 1
               data_dict['total_time'] += duration
               data_dict['avg_time'] = data_dict['total_time']/data_dict['count']
               data_dict['min'] = min(duration, data_dict['min'])
               data_dict['max'] = max(duration, data_dict['max'])
               self.data[func_name] = data_dict
               self.total_time += time.time() - duration - start_count
               self.count += 1
            else:
                result = func(*args, **kwargs)
            return result
        return decorator

class Logger(LoggerABC):
    def __init__(self, loggername="mct"):
        """Write logs to file.

        Args:
            loggername (str, optional): Name of file which is logged. Defaults to "asilla".
        """
        cfg = Config.get_instance()
        level = 'info'
        fmt = '[%(asctime)s][%(levelname)s]:%(message)s'

        if cfg.get_default_bool('debug'):
            level = 'debug' 
            fmt = '[%(asctime)s][%(pathname)s:%(lineno)d][%(levelname)s]:%(message)s'
        stdout = cfg.get_default_bool('stdout')
        formatter = logging.Formatter(fmt = fmt, datefmt = '%F %H:%M:%S')
        log_dir=cfg.get_default('log_dir')
        #default 5M
        max_byte = cfg.get_default_int("max_byte", 5242880) 
        backup_count = cfg.get_default_int("backup_count", 5)
        super().__init__(loggername=loggername, stdout=stdout, log_dir=log_dir, level=level, formatter=formatter, max_byte=max_byte, backup_count=backup_count)

def test():
    logger = Logger()
    logger.info('Running test...')
    timecounters = [TimeInspector().__hash__, TimeInspector().__hash__]
    loggers = [Logger(), Logger()]
    loggers[0].debug(repr(loggers))
    loggers[1].debug(repr(loggers))
    logger.debug(repr((timecounters, loggers)))

if __name__ == '__main__':
    #unittest.main()
    test()

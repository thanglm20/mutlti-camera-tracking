from __future__ import print_function
import os
import configparser
import json
from utils.helper import Singleton, imdict

class ConfigBase(object):
    __name__ = 'ConfigBase'
    def __init__(self):
        self._sub_configs = {}
        pass
 
    def __repr__(self):
        """Return string of _sub_configs.

        Returns:
           str : A string of _sub_configs.
        """
        data = json.dumps(self._sub_configs, indent=4)
        return data
    
    def _get_boolean(self, value):
        """String to boolean.

        Args:
            value (str): Read string.

        Returns:
            bool: Boolean value
        """
        if isinstance(value, str):
            if value.lower() == 'false':
                return False
        return bool(value)
        
    def _get_value(self, opt_name, dtype, default=None):
        """Get value.

        Args:
            opt_name (str): Option's name.
            dtype (dtype): Var's type
            default (any, optional): Default value. Defaults to None.

        Returns:
            dtype: Value of option's name.
        """
        value = self._sub_configs.get(opt_name, None)
        if value is None:
            return default
        assert type(value) in [int, float, str, bool], ValueError(
            'Type Error, cannot get opt: {} | true value: {} | true type: {}'.format(opt_name, value, type(value)))
        return dtype(value)
        
    def items(self):
        """Item from SubConfig

        Returns:
            any: Value of items.
        """
        return self._sub_configs.items()
        
    def getint(self, opt_name, default=None):
        """Return int value.

        Args:
            opt_name (str): Option's name.
            default (any, optional): Default value. Defaults to None.

        Returns:
            int: Int value
        """
        return self._get_value(opt_name, int, default)

    def getfloat(self, opt_name, default=None):
        """Return float value.

        Args:
            opt_name (str):  Option's name.
            default (any, optional): Default value. Defaults to None.

        Returns:
            float: Float value
        """
        return self._get_value(opt_name, float, default)

    def getstr(self, opt_name, default=None):
        """Return str value.

        Args:
            opt_name (str):  Option's name.
            default (any, optional): Default value. Defaults to None.

        Returns:
            str: String value
        """
        return self._get_value(opt_name, str, default)
    
    def getboolean(self, opt_name, default=None):
        """Return boolean value.

        Args:
            opt_name (str):  Option's name.
            default (any, optional): Default value. Defaults to None.

        Returns:
            bool: Boolean value.
        """
        return self._get_value(opt_name, self._get_boolean, default)
    
    def get(self, opt_name, default=None):
        """Return value.

        Args:
            opt_name (str):  Option's name.
            default (any, optional): Default value. Defaults to None.

        Returns:
            any: Value.
        """
        return self._sub_configs.get(opt_name, default)
        
    
class SubConfig(ConfigBase):
    __name__ = 'SubConfig'
    def __init__(self, subname, kwargs):
        super().__init__()
        assert isinstance(kwargs,dict), ValueError('Check kwargs {}'.format(kwargs))
        #print(self, 'init', kwargs)
        self._subname = subname
        self._sub_configs = imdict(kwargs)
        
    def _update(self, kwargs):
        """Update attributes.

        Args:
            kwargs (dict): Key:Value for updating config.
        """
        self._sub_configs = imdict(kwargs)
        
    def __getattribute__(self, name):
        """Get attributes from ConfigBase

        Args:
            name (str): key

        Returns:
            any: value
        """
        try:
            return super().__getattribute__(name)
        except AttributeError:
            raise
    
    def __getattr__(self, name):
        """Get attributes from SubConfig

        Args:
            name (str): SubConfig section names

        Returns:
            dict: Keys:Values of section in SubConfig.
        """
        return self._sub_configs[name]
    
    def __setattr__(self, attribute, val):
        """Set attribute with value.

        Args:
            attribute (str): Attribute's name.
            val (str): Attribute's value.

        Raises:
            TypeError: Object is immutable
        """
        if attribute.startswith("_"):
            super().__setattr__(attribute, val)
        elif attribute in self._sub_configs.keys():
            raise TypeError('Object is Immutable')
    
    def __getitem__(self, index):
        """Get item by index.

        Args:
            index (int): Index of SubConfig's section.

        Returns:
            dict: Keys:Values of section in SubConfig.
        """
        return self._sub_configs[index]
              
class UniqueConfigParser(Singleton):
    '''
    Singleton ConfigParser Class
    UniqueConfigParser is an Immutable Object. It's only set by init or reset function
    
    UniqueConfigParser(...)
        UniqueConfigParser(filepath, kwargs)
            Create an UniqueConfigParser.

            Parameters
            ----------
            filepath: filepath of config file using 'configpaser' format.
            kwargs: if filepath is None, it will initialize as following kwargs
                  e.g:
                  UniqueConfigParser(**{'test':{'test':1,'test2':2}})
        attributes:
            folowing the input config file or kwargs used while initializing or reseting methods
        methods:
            reset(filepath, **kwargs): the same as initializing
    '''
    __name__ = 'UniqueConfigParser'
    __single = True 
    def __init__(self, filepath=None, **kwargs):
        if self.__single:
            super().__init__()
            self.filepath = filepath
            self.kwargs = kwargs
            self._sub_configs = {}
            self.reset(self.filepath, **self.kwargs)
            self.__single = False 

    def reset(self, filepath=None, **kwargs):
        """Reset config

        Args:
            filepath (str, optional): Config path. Defaults to None.
        """
        self.filepath = filepath
        self.kwargs = kwargs
        if self.filepath is None:
            'init by kwargs'
            the_dict = {}
            for key, val in kwargs.items():
                the_dict[key] = val
            #self._configs = imdict(temp)
            # print('ConfigParser is set by key words:', json.dumps(kwargs, indent=4))
        else:
            #'init by configparser'
            assert os.path.exists(filepath),IOError('%s not exist, please check '%self.filepath)
            config = configparser.ConfigParser()
            config.read(filepath, encoding='utf-8')
            the_dict = {}
            for section in sorted(config.sections(), reverse=True):
                temp = {}
                for key, val in config.items(section):
                    temp[key] = val
                the_dict[section] = temp
            the_dict[config.default_section] = config.defaults()
            # print('ConfigParset is set by config file', self.filepath)

        for k,v in the_dict.items():
            sub = self._sub_configs.get(k, None)
            if sub is None:
                self._sub_configs[k] = SubConfig(k, v)
            else:
                sub._update(v)
        
    def __getattribute__(self, name):
        """Get attributes from Config.

        Args:
            name (str): key

        Returns:
            any: value
        """
        try:
            return super().__getattribute__(name)
        except AttributeError:
            raise
    
    def __repr__(self):
        """Return string of _sub_configs.

        Returns:
           str : A string of _sub_configs.
        """
        #print(self._sub_configs)
        #data = json.dumps(self._sub_configs, indent=2)
        data = repr(self._sub_configs)
        return data
    
    def __getattr__(self, name):
        """Get attributes from Config.

        Args:
            name (str): key

        Returns:
            any: value
        """
        try:
            return self._sub_configs[name]
        except KeyError:
            print('There is no such config for {} ==> return EmptyDict'.format(name))
            return SubConfig(name, {})

            
    def __getitem__(self, name):
        """Item from SubConfig

        Args:
            name (str): Section in SubConfig.

        Returns:
            any: Value of items.
        """
        try:
            return self._sub_configs[name]
        except KeyError:
            print('There is no such config for {} ==> return EmptyDict'.format(name))
            return SubConfig(name, {})
    
    @property
    def configs(self):
        """Config

        Returns:
            dict: SubConfigs.
        """
        return self._sub_configs      
           


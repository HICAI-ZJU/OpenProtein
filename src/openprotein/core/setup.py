from typing import *
import weakref
import threading
from collections import UserDict

class Components(type):
    def __new__(cls, name, bases, attrs):
        obj = type.__new__(cls, name, bases, attrs)
        conf = GlobalConfiguration()
        conf.register_component(name, bases, attrs, obj)
        return obj

class _Cached(type):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__cache = weakref.WeakValueDictionary()

    def __call__(self, *args):
        if args in self.__cache:
            return self.__cache[args]
        else:
            obj = super().__call__(*args)
            self.__cache[args] = obj
            return obj

class _Singleton(type):
    def __init__(self, *args, **kwargs):
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.__instance is None:
            self.__instance = super().__call__(*args, **kwargs)
            return self.__instance
        else:
            return self.__instance


class _Restricted(type):
    def __init__(self, *args, **kwargs):
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):

        if self.__instance is None:
            from . import rule
            self.__instance = super().__call__(*args, **kwargs)
            # keys = list(filter(lambda x: "__" not in x, dir(self.rule)))
            attributes = {}
            for key in dir(rule):
                if "__" not in key:
                    attributes["_"+key] = rule.__dict__[key]
                    # self.rule.__dict__.pop(key)
            self.__instance.__dict__.update(attributes)
            return self.__instance
        else:
            return self.__instance

class Tree(UserDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            if key in self.data:
                self.data[key].update(value)
            else:
                value = Tree(value)
                self.data[key] = value
        else:
            self.data[key] = value

    def __getattr__(self, item):
        value = self.data[item]
        if isinstance(value, dict):
            value = Tree(value)
        return value


class GlobalConfiguration(metaclass=_Restricted):

    def __init__(self):
        self._lib = Tree()

    def __getitem__(self, item: str):
        try:
            return self.__dict__[item]
        except KeyError:
            raise AttributeError(f"the {item} is not a valid attribute.")

    def __setitem__(self, key: str, value: str):
        try:
            self.__dict__[key] = value
        except KeyError:
            raise AttributeError(f"the {key} is not a valid attribute.")

    def set_content(self, **kwargs):
        for key, item in kwargs.items():
            try:
                self["_" + key] = item
            except KeyError:
                raise AttributeError(f"the {key} is not a valid attribute.")

    def _inject_properties(self, obj):
        for key, value in self.__dict__.items():
            if isinstance(value, str):
                setattr(obj, key, value)

    def _extract_class_name(self, obj: Union[classmethod, tuple]) -> str:
        if isinstance(obj, tuple):
            name = obj[0].__name__
        else:
            name = obj.__name__
        return name

    def register_component(self, name, bases, attrs, obj):
        self._inject_properties(obj)
        if bases:
            self._lib[self._extract_class_name(bases)][name] = obj
        else:
            self._lib[name] = Tree()


class GlobalUtils(object):
    __slots__ = ('sys', 'logging', 'data', 'train')
    _instance_lock = threading.Lock()

    def __init__(self):
        pass

    def __new__(cls, *args, **kwargs):
        if not hasattr(GlobalUtils, "_instance"):
            with GlobalUtils._instance_lock:
                if not hasattr(GlobalUtils, "_instance"):
                    GlobalUtils._instance = object.__new__(cls)
        return GlobalUtils._instance


from importlib import import_module
import json
import inspect

class Config():
    def __init__(self,filepath) -> None:
        self.conf=self.load_conf(filepath)
        
    @classmethod
    def load_conf(cls,filepath):
        with open(filepath,"r") as fin:
            return json.load(fin)

    def get_transformationlist(self):
        transform_conf=self.conf["augmentation"]["transformation"]
        module_prefix=transform_conf["prefix"]
        T_conf=transform_conf["transforms"]
        transforms=[]
        for item in T_conf:
            class_package=item["name"].split(".")[:-1]
            class_name=item["name"].split(".")[-1]
            module=import_module(module_prefix+"."+".".join(class_package))
            transforms.append(module.__getattribute__(class_name)(**item["param"]))
        return transforms
    
    def get_param(self):
        return self.conf["param"]
    
    def get_func_param(self,func):
        params=inspect.getfullargspec(func)[0]
        return dict((key,value) for key, value in self.conf["param"].items() if key in params)

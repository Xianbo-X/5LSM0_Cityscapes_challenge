from importlib import import_module
import json
import inspect
import os

import torch

class Config():
    """
    Used to parse our configuration file
    """
    def __init__(self,filepath) -> None:
        self.conf=self.load_conf(filepath)
        
    @classmethod
    def load_conf(cls,filepath):
        with open(filepath,"r") as fin:
            return json.load(fin)
    def get_model_name(self):
       return self.conf["architecture"]["model"]
    def get_model(self):
        arch=self.conf["architecture"]
        prefix=arch["prefix"]
        class_package=prefix
        class_name=arch["model"]
        module=import_module(class_package)
        if arch.get("pretrained"):
            print(f"load from {arch['pretrained']}")
            return torch.load(arch["pretrained"])
        return module.__getattribute__(class_name)(**arch["param"])
    def get_lr(self):
        return self.get_param()["learning_rate"]
    
    def get_aug_mode(self):
        return self.get_param()["aug_mode"]
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

    # def get_saves_folder(self):
    #     path = self.conf["path"]
    #     SAVE_DIR = os.path.abspath(path["ROOT_PATH"])
    #     model_folder = os.path.join(SAVE_DIR, path["SAVE_PATH"], path["MODEL_PATH"])
    #     result_folder = os.path.join(SAVE_DIR, path["SAVE_PATH"], path["RESULT_PATH"])

    #     model_path = os.path.join(model_folder, path["MODEL_PREFIX"]+".pt")
    #     return result_folder, model_path
    
    def get_saves_path_prefix(self):
        path = self.conf["path"]
        ROOT_DIR= os.path.abspath(path["ROOT_PATH"])
        model_folder = os.path.join(ROOT_DIR, path["SAVE_PATH"], path["MODEL_PATH"])
        model_prefix = os.path.join(model_folder, path["MODEL_PREFIX"])
        return os.path.abspath(model_prefix)

    def get_save_path(self):
        path = self.conf["path"]
        ROOT_DIR = os.path.abspath(path["ROOT_PATH"])
        return os.path.join(ROOT_DIR,path["SAVE_PATH"])
    
    def get_model_folder(self):
        path = self.conf["path"]
        ROOT_DIR= os.path.abspath(path["ROOT_PATH"])
        model_folder = os.path.join(ROOT_DIR, path["SAVE_PATH"], path["MODEL_PATH"])

        return os.path.abspath(model_folder)

    def get_result_folder(self):
        path = self.conf["path"]
        ROOT_DIR= os.path.abspath(path["ROOT_PATH"])
        result_folder = os.path.join(ROOT_DIR, path["SAVE_PATH"], path["RESULT_PATH"])
        
        return os.path.abspath(result_folder)
        
    def get_architecture(self):
         return self.conf["architecture"]

    def __getitem__(self,key):
        return self.conf[key]
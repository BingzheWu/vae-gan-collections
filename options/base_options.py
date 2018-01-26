
import argparse
import os
class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
    def initialize(self):
        self.parser.add_argument('--dataroot', type = str, required = True, help = 'path to images')
        self.parser.add_argument('--batchSize', type = int, default = 10, help = "input batch size")
        self.parser.add_argument('--imageSize', type = int, default = 224, help = 'input image size')
        self.parser.add_argument('--lr', type = float, default = 0.0001, help = "initial learning rate")
        self.parser.add_argument('--beta1', type = float, default = 0.9, help="hyper-parametr of ADAM")

        self.parser.add_argument('--gpu_ids', type = str, default = '0', help = 'gpu ids: 0,1,2')
        self.parser.add_argument('--name', type = str, default = 'experiments_name', help = 'name of experiments')
        self.parser.add_argument('--checkpoint_dir', type = str, default = './checkpoints', help = "models dir")
        self.parser.add_argument('--is_train', action = 'store_true', help = "if is train")
        self.parser.add_argument('--num_epochs', type = int, default = 10, help = "num of epochs to train")
        self.parser.add_argument('--dataset_name', type = str, default = 'coco_obj_detect', help = "dataset used for training")
        self.parser.add_argument('--annFile', type = str, help = 'path to annotation files')
    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt
    
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os
from yolox.exp import Exp as MyExp
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
#         self.output_dir = "./YOLOX_outputs_test640"
        self.input_size = (2048, 2048)  # (height, width)
        self.test_size = (2048, 2048)
        self.no_aug_epochs = 30
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        # Define yourself dataset path
        self.data_dir = "datasets/traffic5"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
#         self.val_ann = "instances_test2017.json" # 测试的时候解开注释
        self.num_classes = 5
        self.max_epoch = 300
        self.data_num_workers = 4
        self.eval_interval = 1
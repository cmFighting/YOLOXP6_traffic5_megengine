#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import os
import shutil
from loguru import logger

import megengine as mge


# def load_ckpt(model, ckpt):
#     model_state_dict = model.state_dict()
#     load_dict = {}
#     for key_model, v in model_state_dict.items():
#         if key_model not in ckpt:
#             logger.warning(
#                 "{} is not in the ckpt. Please double check and see if this is desired.".format(
#                     key_model
#                 )
#             )
#             continue
#         v_ckpt = ckpt[key_model]
#         if v.shape != v_ckpt.shape:
#             logger.warning(
#                 "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
#                     key_model, v_ckpt.shape, key_model, v.shape
#                 )
#             )
#             continue
#         load_dict[key_model] = v_ckpt

#     for i in range(3):
#         load_dict.pop("head.grids.{}".format(i))
#     model.load_state_dict(load_dict, strict=False)
#     return model
def load_ckpt(model, ckpt):
    model_state_dict = model.state_dict()
    load_dict = {}
    # v是期待的模型
    for key_model, v in model_state_dict.items():

        if key_model not in ckpt:
            logger.warning(
                "{} is not in the ckpt. Please double check and see if this is desired.".format(
                    key_model
                )
            )
            continue
        v_ckpt = ckpt[key_model]
        # 现在需要保证ckpt的能塞进去
        # print(key_model)
        # head.cls_preds.0.bias
        try:
            v_ckpt =v_ckpt.reshape(v.shape)
            if v.shape != v_ckpt.shape:
                logger.warning(
                    "ckpt Shape of {} in checkpoint is {}, while model shape of {} in model is {}.".format(
                        key_model, v_ckpt.shape, key_model, v.shape
                    )
                )
                continue
            load_dict[key_model] = v_ckpt
        except:
            logger.warning(
                "{} is not match".format(key_model)
                )
            # continue
    # for i in range(3):
    #     load_dict.pop("head.grids.{}".format(i))
    model.load_state_dict(load_dict, strict=False)
    return model


def save_checkpoint(state, is_best, save_dir, model_name=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, model_name + "_ckpt.pkl")
    mge.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_dir, "best_ckpt.pkl")
        shutil.copyfile(filename, best_filename)

#!/usr/bin/env python3
# http://weifan-tmm.blogspot.kr/2015/07/a-simple-turorial-for-python-c-inter.html
import argparse
import random
import time
import onnxruntime as ort
import numpy as np
import math
import torch
# 导入torch模块
import torch.nn as nn
import torch.nn.functional as F
import os
import signal
import sys
import sysv_ipc
from type_definitions import *


def signal_handler(sig, frame):
    print('Pressed Ctrl-C!')
    os.system("ipcrm -Q 0x000004d2")
    os.system("ipcrm -Q 0x000010e1")
    sys.exit()

# C发�?36维度feat，接收double
# pyhton发送double，接收feat

FEATURE_SIZE = 40

def receive_from_c(client):
    message, mtype = client["receiver"].receive()
    numpy_message = np.frombuffer(message, dtype=np.double)
    # print(numpy_message)
    # input()

    length = len(numpy_message)
    
    try:
        assert length == FEATURE_SIZE
    except:
        print(FEATURE_SIZE, length)

    # message = numpy_message[0:length]
    return numpy_message


def send_to_c(data, client):
    client["sender"].send(data.tobytes(), block=True, type=TYPE_ARRAY)

def get_s(features, policy_dir):
    # 创建一个推理session，并指定providers参数
    ort_session = ort.InferenceSession(policy_dir, providers=['CPUExecutionProvider'])
    x = np.array(features).astype(np.float32)
    
    outputs = ort_session.run(None,{ 'input' : x})
    res = []
    res.append(0)
    res.append(outputs[0][0])
    res.append(outputs[0][1])
    
    return np.array(res)

def reverse_num(num):
    res_s = str(num)[::-1]
    res = int(res_s)
    if res > 1000000 or res < -1000000:
        res = 0
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--ipc_id',
        help='ipc id',
        type=int,
        default="1234",
    )
    args = parser.parse_args()

    rcv_id = args.ipc_id
    snd_id = reverse_num(rcv_id)

    c_client = {
        "receiver": sysv_ipc.MessageQueue(rcv_id, sysv_ipc.IPC_CREAT),
        "sender": sysv_ipc.MessageQueue(snd_id, sysv_ipc.IPC_CREAT)
    }

    # policy_dir = "/home/xlm/huangxulin/dpl_model/setcover_model_MIX4_mse.onnx"
    # policy_dir = "/home/xlm/huangxulin/dpl_model/setcover_big_Enmodel.onnx" # mix集成学习
    # policy_dir = "/home/xlm/huangxulin/dpl_model/cau_model_mix.onnx"
    # policy_dir = "/home/xlm/huangxulin/dpl_model/cau_enmix_model.onnx"
    # policy_dir = "/home/xlm/huangxulin/dpl_model/fac_enmix_model.onnx"
    # policy_dir = "/home/xlm/huangxulin/dpl_model/setcover_enLTR_model.onnx"
    # policy_dir = "/home/xlm/temp_xlm_node11/dpl_model/setcover_liu_contrast_enmix.onnx"
    policy_dir = "/home/xuliming/xuliming/xuliming/huangxulin/dpl_model/cau_enmix_model.onnx"
    time_total = 0
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
   
    while True:
        receive_feat = receive_from_c(c_client)
        start = time.time()
        # print('node feature: \n', receive_feat)
        out = get_s(receive_feat, policy_dir)
        # print("NN result:", out)
        send_to_c(out, c_client)
        end = time.time()
        time_total += end-start
        print("时间消耗和打分分数：", time_total, out[1], out[2])
        

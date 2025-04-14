#!/usr/bin/env python3
# http://weifan-tmm.blogspot.kr/2015/07/a-simple-turorial-for-python-c-inter.html
import argparse
import os
import signal
import sys
import time
from email import policy
from itertools import groupby

import numpy as np
import sysv_ipc
from type_definitions import *

import xgboost as xgb


def signal_handler(sig, frame):
    print('Pressed Ctrl-C!')
    os.system("ipcrm -Q 0x000004d2")
    os.system("ipcrm -Q 0x000010e1")
    sys.exit()

# C发�?36维度feat，接收double
# pyhton发送double，接收feat
# 每隔一段时间更新模型参�?

FEATURE_SIZE = 20

def receive_from_c(client):
    message, mtype = client["receiver"].receive()
    numpy_message = np.frombuffer(message, dtype=np.double)
    # print(numpy_message)
    # input()

    length = len(numpy_message) - 1
    try:
        assert length == FEATURE_SIZE
    except:
        print(FEATURE_SIZE, length)

    message = numpy_message[0:length]
    numPolicy = numpy_message[length]
    
    return message, numPolicy


def send_to_c(data, client):
    client["sender"].send(data.tobytes(), block=True, type=TYPE_ARRAY)


def load_model(policy_id, policy_dir):
    
    policy_path = os.path.join(policy_dir, f'searchPolicy.{policy_id}.bin')
    model = xgb.Booster({'nthread':2},model_file=policy_path)
    
    return model


def calc_score(input_trj, model):

    # model = xgb.Booster(model_file='searchPolicy.1.bin')
    # predict

    test_list = []
    test_list.append(input_trj)

    test_data = np.asarray(test_list)
    test_data = xgb.DMatrix(test_data)
    rank_score = model.predict(test_data)
    res = []
    res.append(0)
    res.append(rank_score[0])

    return np.array(res)

def get_experiment(experiment):

    if experiment == "":
        ticks = time.localtime(time.time())
        experiment = f'{ticks[0]}0{ticks[1]}{ticks[2]}'
    
    return experiment

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

    # policy_dir = "/home/xuliming/daggerSpace/training_files/scip-dagger/trained_models/setcover/5000_train_1000r_1000c_0.05d/2022-02-24-17-50_nodelist/insL200_trjL8"
    # policy_dir = "/home/xuliming/daggerSpace/training_files/scip-dagger/trained_models/setcover/5000_train_1000r_1000c_0.05d/2022-02-22-11-49_nodelist/train_trj_length1000000_2022-02-24-17"
    policy_dir = "/home/xuliming/daggerSpace/training_files/scip-dagger/trained_models/facilities/train_400_200_100_5/0906_scip3_FSB_oracle_11_heSOL/2022-09-08-23-54_nodelist/insL200_trjL3e6"
    # scip3+afsb+bfs
    # policy_dir = "/home/xuliming/daggerSpace/training_files/scip-dagger/trained_models/indset/train_1000_4/0827_0824_scip3_FSB_oracle_14_heSOL/2022-08-27-10-12_nodelist/insL200_trjL5e7"
    # scip6+learn2branch+bfs
    # policy_dir = "home/xuliming/daggerSpace/training_files/scip-dagger/trained_models/setcover/2500_train_1000r_1000c_0.05d/scip3_afsb_oracle_multi/2022-03-15-15-29_nodelist/insL200_trjL5e4"
    # policy_dir = "/home/xuliming/daggerSpace/training_files/scip-dagger/trained_models/setcover/5000_train_1000r_1000c_0.05d/0829_11_0828_scip3_FSB_oracle_11_heSOL/2022-08-29-10-13_nodelist/insL200_trjL5e6"
    # policy_dir = "/home/xuliming/daggerSpace/training_files/scip-dagger/trained_models/setcover/5000_train_1000r_1000c_0.05d/1103liu_scip6_fsb_oracle_node4_06py/2022-11-04-23-38_nodelist/insL200_trjL5e6"
    # policy_dir = "/home/xuliming/daggerSpace/training_files/scip-dagger/trained_models/cauctions/train2-0_200_1000/1105liu_scip6_fsb_oracle_node4_06py/2022-11-07-10-16_nodelist/insL300_trjL5e7"
    # policy_dir = "/home/xuliming/nodeselector/scip-dagger/policy/cauctions/train2-0_200_1000/0814_liu_hehe_new_scip3_11/searchPolicy.0"
    # policy_dir = "/home/xuliming/daggerSpace/training_files/scip-dagger/trained_models/cauctions/train2-0_200_1000/0819_0817_scip3_afsb_oracle_12_heSOL/2022-08-18-22-40_nodelist/insL200_trjL5e7"
    # policy_dir = "/home/xuliming/daggerSpace/training_files/scip-dagger/trained_models/facilities/train_400_200_100_5/0906_scip3_FSB_oracle_11_heSOL/2022-09-08-23-54_nodelist/insL200_trjL3e6/"
    

    training_files_base = "/home/xuliming/daggerSpace/training_files/scip-dagger"
    
    numPolicy = -1
    time_total = 0
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
   
    while True:

        receive_feat, new_numPolicy = receive_from_c(c_client)
        # receive_feat = [1.008382, 1.201734, 0.0, 1.0, 0.0, -0.364382, 0.0, 0.0, 0.0, 2.519651, 0.0, 0.050779, 1.165065, 0.165065, 0.0, 0.0, 0.0, 0.01, 0.0, 1.0]
        new_numPolicy = 1
        start = time.time()
        # print("policy ", int(numPolicy), int(new_numPolicy))
        # print('node feature: \n', receive_feat)

        if numPolicy != int(new_numPolicy):
            
            numPolicy = new_numPolicy
            model = load_model(int(new_numPolicy), policy_dir)
            # print(f'Get a new policy: {policy_dir}searchPolicy.{int(numPolicy)}.bin')

        out = calc_score(receive_feat, model)

        # print("NN result:", out)
        send_to_c(out, c_client)
        end = time.time()
        time_total += end-start
        print("时间和打分：", time_total, out)

# ====================================================
# Collect all nodes and write in a file
# ====================================================
from audioop import avg, reverse
from copyreg import pickle
from posixpath import split
import random
import itertools
import argparse
import datetime
import os
import math
import json
import pickle
import logging
import datetime
from re import A
from utils import *
from get_scip_node import InstanceFile
from get_scip_node import Node

class TrjDataMulti():
    # instance 
    def __init__(self, trj_file_path, instance_file_obj, topk=100):
        self.lines = open("%s" % trj_file_path).readlines()
        self.instance_file_obj = instance_file_obj
    
    def cut(self, obj, sec):

        return [obj[i:i+sec] for i in range(0,len(obj),sec)]
        
    def node_check_optimal(self, node_idx):

        optID = 0
        is_opt = False
        cur_node = self.instance_file_obj.sorted_nodelist[node_idx]
        
        try:
            cur_branchvars = cur_node.branchvars
        except:
            print("node_check_optimal wrong:", node_idx, len(self.instance_file_obj.sorted_nodelist), self.instance_file_obj.max_node_number)
        
        m = len(cur_branchvars)
        opt_node_list = self.instance_file_obj.best_primalbound_nodelist
        opt_node = opt_node_list[len(opt_node_list) - 1]
        if abs(opt_node.sol - self.instance_file_obj.best_primalbound) < 1:
            opt_branchvars = opt_node.branchvars
            n = len(opt_branchvars)

            if m > n: 
                is_opt = False
                pass

            elif m == 0: 
                is_opt = True
            else:
                for i in range(len(cur_branchvars)):
                    if is_same_var(cur_branchvars[m-i-1], opt_branchvars[n-i-1]):
                        is_opt = True
                        continue
                    else:
                        is_opt = False
                        break
                if is_opt == True:
                    optID = len(opt_node.branchvars)

        node_label = {}
        node_label["value"] = 1 if is_opt else 0
        node_label["optID"] = optID
        return node_label

    def gen_pqlist(self, line, group_id=0, is_pure=True):
        FEATURE_SIZE = 22
        line = line.strip(" \n").split(" ")
        try:
            feats = [float(a.split(":")[1]) for a in line] # 22 * nnodes
        except:
            logging.info("gen_pqlist %s: ", list_to_str(line))
            return -1
        try:
            assert len(feats) % FEATURE_SIZE == 0
        except:
            logging.info("gen_pqlist %s %d %d " % (list_to_str(feats), len(feats), FEATURE_SIZE))
        pqlist = []
        cutted_feats = self.cut(feats, FEATURE_SIZE)
        for feat in cutted_feats:
            if int(feat[0]) >= self.instance_file_obj.max_node_number:
                continue
            
            # nodeID, groupID, feats(*20)
            try:
                record_node = {}
                record_node["idx"] = int(feat[0])
                record_node["groupID"] = int(feat[1]) # 暂时不用，但先收�?
                record_node["feats"] = feat[2:]
                record_node["label"] = self.node_check_optimal(record_node["idx"])
                pqlist.append(record_node)
            except:
                print("TrjDataMulti-gen_pqlist: wrong feats ")
                print(feat)
                input()
                continue

        return pqlist
    
    def write_list_to_json(self, nodelist, json_file_name, json_file_save_path):

        tmp_file_path = os.path.join(json_file_save_path, json_file_name)
        with open(tmp_file_path, 'a') as f:
            tmp = json.dumps(nodelist)
            f.write("%s\n" % tmp)
            f.close()
    
    def write_list_to_pickle(self, nodelist, pickle_file_name, pickle_file_save_path):
        tmp_file_path = os.path.join(pickle_file_save_path, pickle_file_name)
        with open(tmp_file_path, 'ab') as f:
            pickle.dump(nodelist, f)
            f.close()
    
    def extract_trjs(self, files_dict, save_path):
        # One pq_list
        pqlists = [self.gen_pqlist(line) for line in self.lines if self.gen_pqlist(line) != -1] 
        records = list(itertools.chain.from_iterable(pqlists))
        self.write_list_to_json(records, files_dict["json"], save_path)
        self.write_list_to_pickle(records, files_dict["pickle"], save_path)
        
def get_all_trjs_to_json(dat_dir, trj_dir, log_dir, files_name, files_save_path, first_k = 5000):

    tmp_list = []
    tmp_list_gt1 = []
    instances_list = sorted(os.listdir(dat_dir), key=lambda x:int(x.split('.')[0].split('_')[1]))
    print("get_all_trjs_to_json")

    for i in range(0, len(instances_list)):
        instance = instances_list[i]
        if i > 100:
            continue
        print("%i %s" %(i, instance))
        base = instance.split(".")[0]
        trj_path = os.path.join(trj_dir, base+".trj") 
        log_path = os.path.join(log_dir, base+".log")
        if os.path.exists(trj_path) == False or os.path.exists(log_path) == False:
            continue
        ins_file_obj = InstanceFile(log_path)
        bPB_time = ins_file_obj.bPB_time
        presolving_time = ins_file_obj.presolving_time
        if bPB_time - presolving_time <= 0.1:
            print("presoled BPB.", bPB_time, presolving_time)
            continue
        best_primalbound_nodelist_len = 0

        try:
            instance_file_obj = InstanceFile(log_path,dat_type)
            logging.info("%i %s" %(i, instance))
            best_primalbound_nodelist_len = len(instance_file_obj.best_primalbound_nodelist)
            tmp_list.append(best_primalbound_nodelist_len)
            tmp_sols = [node.sol for node in instance_file_obj.best_primalbound_nodelist]
            logging.info("tmp_sols %d %s" % (best_primalbound_nodelist_len, list_to_str(tmp_sols)))
            print("tmp_sols %d [%s ]" % (best_primalbound_nodelist_len, list_to_str(tmp_sols)))
            
            if best_primalbound_nodelist_len > 1:
                tmp_list_gt1.append(best_primalbound_nodelist_len)
        except:
            print("get_all_trjs_to_json: InstanceFile wrong && continue")

        if best_primalbound_nodelist_len > 0:
            trj_data_multi = TrjDataMulti(trj_path, instance_file_obj)
            trj_data_multi.extract_trjs(files_name, files_save_path)
        
    try:
        logging.info("mean all sols length %d" % int(sum(tmp_list)/len(tmp_list)))
    except:
        logging.info("all sols sum %d len %d" % (sum(tmp_list), len(tmp_list)))
    try:
        logging.info("mean all sols2 length %d", int(sum(tmp_list_gt1)/len(tmp_list_gt1)))
    except:
        logging.info("all sols2 sum %d len %d" % (sum(tmp_list), len(tmp_list_gt1)))



if __name__ == "__main__":

    random.seed(43211234)
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    parser = argparse.ArgumentParser()
    parser.add_argument(
       '-t', '--dat_type',
       help='data type',
       type=str,
       default='setcover',
    )
    parser.add_argument(
        '-d', '--data',
        help='input data dir',
        type=str,
        default="",
    )
    parser.add_argument(
       '-e', '--experiment',
       help='experiment name',
       type=str,
       default=f'tmp_{now_time}',
    )
    parser.add_argument(
       '-k', '--first_k',
       help='experiment name',
       type=int,
       default=20,
    )

    # setcover
    dat_type = "setcover"
    dat_name = "5000_train_1000r_1000c_0.05d"
    experiment = "0827xulm_scip6_fsb_oracle_node11" #TODO
    dat_dir_base = f'./instance_data/{dat_type}'
    log_dir_base = f'./training_data/result/{dat_type}'
    trj_dir_base = f'./training_data/trj/{dat_type}'

    dat_dir = os.path.join(dat_dir_base, dat_name)
    trj_dir = os.path.join(trj_dir_base, dat_name, experiment)
    log_dir = os.path.join(log_dir_base, dat_name, experiment)

    log_file_name = "./make_data_files/collect_trjs_logs/" + now_time + ".log"
    train_file_save_path = os.path.join(trj_dir, "train_pickle_json")
    os.makedirs(train_file_save_path, exist_ok=True)
    
    train_files_save_name = {}
    json_file_name = now_time + "_nodelist.json"
    pickle_file_name = now_time + "_nodelist.pickle"
    train_files_save_name["json"] = json_file_name
    train_files_save_name["pickle"] = pickle_file_name
    
    logging.basicConfig(
        filename=log_file_name,
        format='%(asctime)s-%(name)s-%(levelname)s-%(module)s:%(message)s',
        level=10,
        filemode="w",
        datefmt='%m/%d/%Y %I:%M:%S'
    )
    print(train_files_save_name)
    print(train_file_save_path)

    get_all_trjs_to_json(dat_dir, trj_dir, log_dir, train_files_save_name, train_file_save_path)

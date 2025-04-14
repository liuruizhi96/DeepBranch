import os
import math
import json
import pickle
import argparse
import datetime
from matplotlib.pyplot import flag
import numpy as np
import xgboost as xgb
from itertools import groupby
import pdb
# ins 28
def read_json(json_file):

    print("read json")
    total_pairs = []
    with open(json_file, 'r', encoding="utf-8") as f:
        line = f.readline()
        while line:
            instance_trjs = json.loads(line.strip("\n"))
            print(len(instance_trjs), instance_trjs[0])
            line = f.readline()
    f.close()
    return total_pairs

def make_pairwise_data(trjs):


    total_label1 = sum([1 for trj in trjs if trj["label"]["value"] == 1])
    if total_label1 == 0:
        print("tmp debug: make_pairwise_data, empty")

        return [], [], []

    rank_feats = []
    rank_group = []
    rank_label = []
    
    length = 0
    sum_trjs = 0
    sum_label1 = 0
  
    i = 0
    for group_id, pq_trjs in groupby(trjs, lambda x: x["groupID"]):
        i += 1
        pre_length = length
        length = 0
        sum_label1 = 0
        cur_feats = []
        cur_label = []
        for trj in pq_trjs:

            length += 1
            sum_trjs += 1
            cur_feats.append(trj["feats"])
            cur_label.append(trj["label"]["value"])
            sum_label1 += 1 if trj["label"]["value"] == 1 else 0

        if sum_label1 != 0 and (i%5 == 0):
            rank_feats.extend(cur_feats)
            rank_label.extend(cur_label)
            rank_group.append(length)


    return rank_feats, rank_label, rank_group

def read_pickle_train(pickle_file, trained_models_path):

    print("read pickle")

    sum_ins = 0
    flag = True
    train_iter = 0
    sum_train_ins = 0

    test_ins_length = 100   
    train_ins_length = 100  
    test_trj_length = 5e3   
    train_trj_length = 5e3  
    
    pre_model_path = ""

    test_rank_feats = []
    test_rank_label = []
    test_rank_group = []

    batch_train_rank_feats = []
    batch_train_rank_label = []
    batch_train_rank_group = []

    iters = 50

    model = xgb.XGBRanker(
        booster='gbtree',
        objective='rank:pairwise',
        random_state=42,
        learning_rate=0.05,
        colsample_bytree=0.9,
        eta=0.05,
        max_depth=6,
        n_estimators=iters,
        subsample=0.75,
        verbosity=0
        )
    
    pickle_file_name = pickle_file.split('/')[-1].split('.')[0]
    cur_model_dir = os.path.join(
        trained_models_path, 
        pickle_file_name, 
        "insL"+str(train_ins_length)+\
        "_trjL"+str(train_trj_length)[0]+\
        'e'+str(int(math.log10(train_trj_length)))
        )
    
    if os.path.isdir(cur_model_dir) == False:
        os.makedirs(cur_model_dir, exist_ok=False)
    iter = 0
    with open(pickle_file, 'rb') as f:
        while True:
            print("iter %d debug" % iter)
            iter += 1
            try:
                instance_trjs = []
                instance_trjs = pickle.load(f)
                ins_feats, ins_label, ins_group = make_pairwise_data(instance_trjs)
                
                try:
                    assert len(ins_feats) != 0
                except:
                    print("empty or negative trjs")
                    continue
            except EOFError:
                break

            sum_ins += 1
            if sum_ins < test_ins_length and len(test_rank_feats) < test_trj_length:
                test_rank_feats.extend(ins_feats)
                test_rank_label.extend(ins_label)
                test_rank_group.extend(ins_group)
                print("test: cur_ins(%d) total(%d) cur(%d)" % \
                        (sum_ins, len(test_rank_feats), len(ins_feats) ))
            else:
                if flag == True:
                    flag = False

                    test_X, test_Y, test_group =   \
                        np.array(test_rank_feats), \
                        np.array(test_rank_label), \
                        test_rank_group
                    
                sum_train_ins += 1
                batch_train_rank_feats.extend(ins_feats)
                batch_train_rank_label.extend(ins_label)
                batch_train_rank_group.extend(ins_group)
                print("train: iter(%d) cur_ins(%d) cur_ins(%d) total(%d) cur(%d)" % \
                        (train_iter, sum_ins, sum_train_ins, len(batch_train_rank_feats), len(ins_feats) ))

            if sum_train_ins == train_ins_length or len(batch_train_rank_feats) > train_trj_length:
                model_name = "searchPolicy."+str(train_iter) + ".bin"
                cur_model_path = os.path.join(cur_model_dir, model_name)
                
                train_X, train_Y, train_group =         \
                    np.array(batch_train_rank_feats),   \
                    np.array(batch_train_rank_label),   \
                    batch_train_rank_group
    
                try:
                    if pre_model_path == "":
                        model.fit(
                            train_X, 
                            train_Y, 
                            group=train_group, 
                            verbose=True, 
                            eval_set=[(train_X, train_Y),(test_X, test_Y)], 
                            eval_group=[train_group,test_group]
                        )
                    else:
                        model.fit(
                            train_X, train_Y, 
                            group=train_group, 
                            verbose=True, 
                            eval_set=[(train_X, train_Y),(test_X, test_Y)], 
                            eval_group=[train_group,test_group], 
                            xgb_model=pre_model_path
                            )
                except:
                    print("mode.fit error")
                model.save_model(cur_model_path)

                train_iter += 1
                sum_train_ins = 0
                pre_model_path = cur_model_path
                batch_train_rank_feats = []
                batch_train_rank_label = []
                batch_train_rank_group = []
        f.close()
    
if __name__ == "__main__":
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
        default='2500_train_1000r_1000c_0.05d',
    )
    parser.add_argument(
       '-e', '--experiment',
       help='experiment name',
       type=str,
       default='scip3_afsb_oracle_multi',
    )
    parser.add_argument(
       '--train_file_path',
       help='train_file_path (pickle path)',
       type=str,
       default='',
    )
    parser.add_argument(
       '-k', '--first_k',
       help='experiment name',
       type=int,
       default=20,
    )
    
    # scip6 + baseline + bfs
    training_base = "/home/xuliming/daggerSpace/training_files/scip-dagger"
    args = parser.parse_args()
    dat_name = args.data
    dat_type = args.dat_type
    experiment = args.experiment
    train_file_path = args.train_file_path
    trained_model_path = os.path.join(
        training_base,
        "trained_models",
        dat_type, 
        dat_name,
        experiment
        )
    print(trained_model_path)

    read_pickle_train(train_file_path, trained_model_path)

import os
import argparse
import datetime
import signal
import sys

def signal_handler(sig, frame):
    print('Pressed Ctrl-C!')
    sys.exit()

def run_diff_policy(dat_dir, result_dir, fk_file_list):
    signal.signal(signal.SIGINT, signal_handler)
    print("Dataset: ", dat_dir)
    print("Log: ", result_dir)
    for j, ins_file in enumerate(fk_file_list):        
        base = ins_file.split('.lp')[0]
        f_path = os.path.join(dat_dir, ins_file)
        r_policy_dir = os.path.join(result_dir, f'policy.{0}')
        if os.path.isdir(r_policy_dir) == False:
            os.makedirs(r_policy_dir, exist_ok=False)
        r_path = os.path.join(r_policy_dir, f'{base}.log')
        try:
            os.system(f"bin/scip6_dl_set_pair -s ./sets/fullstrong_bfs.set -f %s > %s" % (f_path, r_path))
        except KeyboardInterrupt:
            print("numpolicy %d %d %s stopped" % (0, j, ins_file))
            continue


def run_scip(dat_dir, sol_dir, result_dir, set_path, exe="collect_multi_trjs_new", timelimit=3600, first_k=20, action="scip"):
    signal.signal(signal.SIGINT, signal_handler)
    file_list = sorted(os.listdir(dat_dir), key=lambda x:int(x.split('.')[0].split('_')[1]))    
    print(sol_dir)
    print(result_dir)
    if os.path.isdir(result_dir) == False:
        os.makedirs(result_dir, exist_ok=False)
    for j, ins_file in enumerate(file_list):
        try:
            base = ins_file.split('.')[0]
            print("%d %s" % (j, base))
            if j >= first_k:
                continue
            f_path = os.path.join(dat_dir, ins_file)
            s_path = os.path.join(sol_dir, f'{base}.sol')
            r_path = os.path.join(result_dir, f'{base}.log')
            if action == "scip":
                os.system("bin/scip -t %d -f %s -s %s > %s" % (timelimit, f_path, set_path, r_path))
            else:
                os.system("bin/0427 -t %d -f %s -s %s -o %s --nodesel oracle > %s" % (timelimit, f_path, set_path, s_path, r_path))
        except KeyboardInterrupt:
            sys.exit()


if __name__ == "__main__":

    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    parser = argparse.ArgumentParser()
    parser.add_argument(
       '-a', '--action',
       help='action',
       choices=['policy', 'scip', 'oracle'],
    )
    parser.add_argument(
       '-t', '--dat_type',
       help='data type',
       type=str,
       default='setcover',
    )
    parser.add_argument(
        '-d', '--dat_name',
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
       '-s', '--set',
       help='set name',
       type=str,
       default=f'./sets/fullstrong_bfs.set',
    )
    parser.add_argument(
       '-k', '--first_k',
       help='first k',
       type=int,
       default=120,
    )
    
    args = parser.parse_args()
    dat_type = "setcover"
    fk = args.first_k
    set_name = args.set
    dat_name = args.dat_name
    dat_type = args.dat_type
    experiment = args.experiment
    training_files_base="./training_data"
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    d_path = os.path.join(training_files_base, "dat",dat_type, dat_name)
    s_dir = os.path.join(training_files_base, "bfs_solution", dat_type, dat_name)
    r_dir = os.path.join(training_files_base, "result", dat_type, dat_name, experiment)
    file_list = sorted(os.listdir(d_path), key=lambda x:int(x.split('.')[0].split('_')[1])) 
    fk_file_list = file_list[:100]
    if args.action == 'policy':
        run_diff_policy(dat_dir=d_path, result_dir=r_dir, fk_file_list=fk_file_list)
    else:
        run_scip(dat_dir=d_path, sol_dir=s_dir, result_dir=r_dir, set_path=set_name, first_k=fk, action=args.action)

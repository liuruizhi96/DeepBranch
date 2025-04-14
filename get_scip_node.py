import os

class Node():
    def __init__(self, idx=-2, lowerbound=-1.00, primalbound=10000.00, dualbound=-1.00, branchvars=[], left=0, time=0.0, depth=0, obj_list=[10000]):
        self.idx = idx
        self.lowerbound = lowerbound
        self.primalbound = primalbound
        self.dualbound = dualbound
        self.branchvars = branchvars   
        self.left = left               
        self.time = time                
        self.depth = depth
        self.obj_list = obj_list
        self.sol = 0.00
        self.bPB_time = 0.00
        self.nvars = 0
        self.varstime = 0.00

class InstanceFile():
    def __init__(self, log_file_path="", dat_type = 'facilities'):
        self.bPB_time = 0.0
        self.log_file_lines = []
        self.max_node_number = 0
        self.presolving_time = 0
        # Initialize best_primalbound according to diff prob types
        if dat_type == 'cauctions':
            self.best_primalbound = 10  
        if dat_type == 'facilities' or dat_type == 'setcover':
            self.best_primalbound = 100000000  
        self.nodelist = self.get_all_nodes_from_log(log_file_path,dat_type)
        self.accnodelist = self.get_all_nodes_ACC_from_log(log_file_path,dat_type)

        self.bPB_time = self.get_bPB_time()
        print('now: ', self.bPB_time)
        self.sorted_nodelist = self.build_sorted_nodelist()

        self.presolving_time = self.get_presolving_time()
        self.best_primalbound_nodelist = self.get_multi_bPB() 

    def is_leaves(self, *nodes):
        for node in nodes:
            if node.idx == 1 or node.idx == -1:
                return False
        return True
    
    def get_presolving_time(self):
        if len(self.sorted_nodelist) <= 4:
            return self.sorted_nodelist[self.max_node_number].time
        else:
            return min(self.sorted_nodelist[2].time, self.sorted_nodelist[3].time)

    def get_bPB_time(self):

        b_time = -1.0
        try:
            b_time = self.nodelist[-1].bPB_time

        except:
            print("empty bPB")
        
        return b_time
        
    def get_all_nodes_from_log(self, log_file_path,dat_type): 
        os.system("grep -rEn 'final selecting' %s | cut -f1 -d':' > %s_tmp" % (log_file_path, log_file_path))
        line_no = [int(n.strip("\n")) for n in open("%s_tmp" % log_file_path, "r").readlines()] # 1,2,...
        os.system("rm %s_tmp" % log_file_path)
        self.log_file_lines = open("%s" % log_file_path, "r").readlines()

        cur_nodelist = []
        if dat_type == 'cauctions':
            for i in range(len(line_no)):
                cur_node = self.get_scip_node(line_no[i-1], line_no[i])
                if cur_node != Node() and cur_node.primalbound != -1:
                    cur_nodelist.append(cur_node)
                    self.best_primalbound = max(cur_node.primalbound, self.best_primalbound)
                    self.max_node_number = max(cur_node.idx, self.max_node_number)
        
        if dat_type == 'facilities' or dat_type == 'setcover':
            for i in range(len(line_no)):
                cur_node = self.get_scip_node(line_no[i-1], line_no[i])
                if cur_node != Node() and cur_node.primalbound != -1:
                    cur_nodelist.append(cur_node)
                    self.best_primalbound = min(cur_node.primalbound, self.best_primalbound)
                    self.max_node_number = max(cur_node.idx, self.max_node_number)

        return cur_nodelist
    
### LIU TOP-ACC 1024
    def get_all_nodes_ACC_from_log(self, log_file_path,dat_type): 
        os.system("grep -rEn 'opt is better' %s | cut -f1 -d':' > %s_tmp" % (log_file_path, log_file_path))
        line_no = [int(n.strip("\n")) for n in open("%s_tmp" % log_file_path, "r").readlines()] # 1,2,...
        os.system("rm %s_tmp" % log_file_path)
        self.log_file_lines = open("%s" % log_file_path, "r").readlines()

        cur_nodelist = []
        if dat_type == 'cauctions' or dat_type == 'indset' or dat_type == 'GISP':
            for i in range(len(line_no)):
                cur_node = self.get_scip_node(line_no[i-1], line_no[i])
                if cur_node != Node() and cur_node.primalbound != -1:
                    cur_nodelist.append(cur_node)
                    self.best_primalbound = max(cur_node.primalbound, self.best_primalbound)
                    self.max_node_number = max(cur_node.idx, self.max_node_number)
        
        if dat_type == 'facilities' or dat_type == 'setcover'or dat_type == 'MIK':
            for i in range(len(line_no)):
                cur_node = self.get_scip_node(line_no[i-1], line_no[i])
                if cur_node != Node() and cur_node.primalbound != -1:
                    cur_nodelist.append(cur_node)
                    self.best_primalbound = min(cur_node.primalbound, self.best_primalbound)
                    self.max_node_number = max(cur_node.idx, self.max_node_number)
                
        return cur_nodelist
    
    def get_new_sols(self, node1, node2):
        # node1 - node2
        obj_list_1 = node1.obj_list
        obj_list_2 = node2.obj_list
        
        diff = []
        obj_hash = {}

        for i, j in zip(obj_list_1, obj_list_2):
            if i in obj_hash.keys():
                obj_hash[i] += 1
            else:
                obj_hash[i] = 1
        for j in obj_list_2:
            if j in obj_hash.keys():
                obj_hash[j] -= 1
        for key in obj_hash:
            if obj_hash[key] >= 1:
                diff.extend([key]*int(obj_hash[key]))
        
        try:
            assert len(diff) == 1
            return diff[0]
        except:
            print("get_new_sols wrong")
            return min(diff)
        
    def get_multi_bPB(self):
        improvment_nodelist = []
        for i in range(1, len(self.nodelist)):
            
            if len(self.nodelist[i].obj_list) == 0:
                continue
            
            cur_obj = self.nodelist[i].obj_list
            pre_obj = self.nodelist[i-1].obj_list
            try:
                if self.nodelist[i].primalbound == self.best_primalbound and self.nodelist[i-1].primalbound != self.best_primalbound:
                    self.nodelist[i-1].sol = cur_obj[0]
                    improvment_nodelist.append(self.nodelist[i-1])
            except:
                print(len(cur_obj), len(pre_obj))
                print(cur_obj)
                print(pre_obj)
                input()
        
        return improvment_nodelist

    def get_scip_node(self, st, ed):

        node = Node()
        _splitted = self.log_file_lines[ed-1].strip().rstrip("\n").split()

        if _splitted[0] == "1:":
            splitted = _splitted[1:]
        else:
            splitted = _splitted
        idx = splitted[6]

        if idx == -1 or idx == 1:
            return node
       
        node.idx = int(idx)
        node.primalbound = float(splitted[8])
        node.lowerbound = float(splitted[10])
        node.dualbound = float(splitted[12])

        if len(splitted) > 13:
            node.time = float(splitted[14])
            node.depth = int(splitted[16])
            node.left = int(splitted[18])
            try:
                node.bPB_time = float(splitted[20])
            except:
                node.bPB_time = 0
        if len(splitted) > 20:
            node_obj_list = []
            for i in range(25, len(splitted), 1):
                if "obj" in splitted[i-1]:
                    node_obj_list.append(float(splitted[i]))
            node.obj_list = node_obj_list
            
        node_branchvars = []
        for i in range(ed-2, st-2, -1):
            f_l = self.log_file_lines[i].strip("\n")
            if f_l.startswith("<"):
                try:
                    t_var, rel, value = f_l.split(" ")
                except:
                    print("[bad t_var], %s" % f_l)
                var = {}
                if rel == "<=" and value == "0.0":
                    var["name"] = self.process_t_var(t_var)
                    var["value"] = int(0)
                elif rel == ">=" and value == "1.0":
                    var["name"] = self.process_t_var(t_var)
                    var["value"] = int(1)
                if var:
                    node_branchvars.append(var)
            elif f_l.startswith("[src/scip/nodesel"):
                break
            else:
                continue
            
        node.branchvars = node_branchvars
        return node
    
    # Sort by node number
    def build_sorted_nodelist(self):

        sorted_nodelist = [Node()]*(self.max_node_number+1)
        for node in self.nodelist:
            try:
                sorted_nodelist[node.idx] = node
            except:
                print("Wrong, build_sorted_nodelist %d %d %d" %( len(sorted_nodelist), node.idx, self.max_node_number))
        
        return sorted_nodelist

    def process_t_var(self, t_var):
        assert t_var[0] == "<"
        assert t_var[-1]== ">"
        tv = t_var[1:-1]
        while tv.startswith("t_"):
            tv = tv.replace("t_", "")
        return tv


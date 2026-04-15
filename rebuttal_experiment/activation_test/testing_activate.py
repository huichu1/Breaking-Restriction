from deep_layer_utils import *
import numpy as np
import torch
result_file = open("testing_activation.txt","w")

def Checking_activation():
    num = 0
    duals = []
    root = FOLDER_NAME  + EXP_ROOT_FOLDER_NAME + "/list/"
    for f in sorted(os.listdir(root)):
        print(f)
        x = pickle.load(open(os.path.join(root,f),"rb"))
        duals.extend(x)
    all_points_num = len(duals)
    print(f"all points num is {all_points_num}",file=result_file)
    result_list = []
    for i,(left,mid,right) in enumerate(duals):
        if i%1000 == 0:
            print(f"ieration {i}/{all_points_num}")
        
        result = cheat_cuda(torch.from_numpy(mid).cuda().double())
        result = result.cpu().numpy()
        result = np.where(result > 0,1,0)
        result = result.reshape((12,64))        
        result_list.append(result)
    result_list = np.array(result_list)
    sum_result = np.sum(result_list,axis=0)
    print(sum_result,file=result_file)
    sum_result = sum_result / all_points_num
    print(sum_result,file=result_file)

    min_result = np.min(sum_result,axis = 1)
    max_result = np.max(sum_result,axis = 1)
    print(min_result,file=result_file)
    print(max_result,file=result_file)
    
Checking_activation()
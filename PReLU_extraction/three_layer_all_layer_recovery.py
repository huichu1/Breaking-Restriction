import re
import os
import pickle
from three_layer_utils import *
from collections import defaultdict
import sys
from scipy.optimize import least_squares

tmp_file = open("./process.txt","w")

have_null_space = False
pre_null_space = None
soln_order = []

def intersect(left, right, nleft, nright):
    A = np.vstack((nleft, nright))
    b = np.array([np.dot(nleft, left), np.dot(nright, right)])

    # Find a particular solution
    x0 = np.linalg.lstsq(A, b, rcond=None)[0]    
    # Find the null space of A
    N = scipy.linalg.null_space(A, 1e-5)
    
    return x0, N

# Function to generate random points on the n-2 dimensional subspace
def generate_points_on_subspace(x0, N, num_points=10):
    random_vectors = np.random.randn(N.shape[1], num_points)
    random_vectors = random_vectors/3000
    subspace_points = x0[:, np.newaxis] + N @ random_vectors
    return subspace_points.T

def generate_points_on_subspace_stepsize(x0, N, num_points=10,step_size = 3000):
    random_vectors = np.random.randn(N.shape[1], num_points)
    random_vectors = random_vectors/step_size
    subspace_points = x0[:, np.newaxis] + N @ random_vectors
    return subspace_points.T


def vectorized_check_closest_pair_distance(points):
    # Extract the second coordinate from each point
    coords = np.array([p[1] for p in points])
    
    # Calculate pairwise distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distances = np.sum(np.square(diff), axis=-1)
    
    # Set diagonal to infinity to ignore self-distances
    np.fill_diagonal(distances, -np.inf)
    
    # Find the minimum distance
    min_distance = np.max(distances)
    
    if min_distance < 1:
        return True
    else:
        return False

class CIFAR10NetPrefix(nn.Module):
    def __init__(self, layers):
        super(CIFAR10NetPrefix, self).__init__()
        if layers == 0:
            self.fcs = nn.Sequential()
            self.layers = 0
        else:
            self.layers = layers
            h = [DIM1,DIM2, SHRINK,10]
            new_layers = [nn.Linear(h[layer], h[layer+1]) for layer in range(layers-1)]
            self.fcs = nn.Sequential(*([nn.Linear(IDIM, DIM1)] + new_layers))
            prelu_layers = [nn.PReLU(h[layer+1]) for layer in range(layers-1)]
            self.relu = nn.ReLU()
            self.prelus = nn.Sequential(*([nn.PReLU(DIM1)] + prelu_layers))

        self.double()

    def prelu_around(self, x, layer):
        # print("test")
        # print(x[:1] >= 0)
        this_layer_prelu_index = self.prelus[layer]
        prelu_weight = this_layer_prelu_index.weight
        # print(prelu_weight)
        
        prelu_weight = torch.unsqueeze(prelu_weight,0)
        prelu_weight[x[:1] >= 0] = 1
        
        return x * prelu_weight
        
    @torch.no_grad
    def forward_around(self, x):
        x = x.view(-1, IDIM)
        if len(self.fcs) == 0: return x
        for i in range(len(self.fcs)):
            x = self.prelu_around(self.fcs[i](x),i)
        return x

    @torch.no_grad
    def forward(self, x):
        x = x.view(-1, IDIM)
        if len(self.fcs) == 0: return x
        for i in range(len(self.fcs)):
            x = self.prelus[i](self.fcs[i](x))
            # print(x)
        return x

def transfer_weights(source_model, target_model, source_prefix='', target_prefix='fcs'):
    target_state_dict = {}
    source_state_dict = source_model.state_dict()
    layer_count = 0
    while True:
        source_weight_key = f'{source_prefix}fc{layer_count+1}.weight'
        source_bias_key = f'{source_prefix}fc{layer_count+1}.bias'
        if layer_count != 3:
            source_slope_key = f'{source_prefix}prelu{layer_count+1}.weight'
        
        if source_weight_key not in source_state_dict:
            break
        
        target_weight_key = f'{target_prefix}.{layer_count}.weight'
        target_bias_key = f'{target_prefix}.{layer_count}.bias'
        target_slope_key = f'prelus.{layer_count}.weight'

        target_state_dict[target_weight_key] = source_state_dict[source_weight_key]
        target_state_dict[target_bias_key] = source_state_dict[source_bias_key]
        target_state_dict[target_slope_key] = source_state_dict[source_slope_key]
        layer_count += 1

    target_model.load_state_dict(target_state_dict, strict=False)

    return layer_count
    
def extend_coeff(coeff,layer):
    new_part_coeff = np.zeros((coeff.shape[0],WEIGHTS_DIM_LIST[layer+1]))
    new_part_coeff[:,0:WEIGHTS_DIM_LIST[layer]] = coeff
    for i in range(len(coeff)):
        new_part_coeff[i][i + WEIGHTS_DIM_LIST[layer]] = -1
    new_coeff = np.zeros((coeff.shape[0],WEIGHTS_DIM_LIST[layer+1]*2))
    new_coeff[:,0:WEIGHTS_DIM_LIST[layer+1]] = new_part_coeff
    new_coeff[:,WEIGHTS_DIM_LIST[layer+1]:] = new_part_coeff
    return new_coeff 

def is_consistent_help(points, prefix, layer=0, do_return_soln=False, allow_close=False,step_size = 3000):
    print("same cluster begin!",file=RESULT_FILE)
    global have_null_space
    global pre_null_space
    global soln_order
    samples = []
    global tmp_file
    target_step_size = step_size
    # The points need to be in different linear regions to try and compare them
    if vectorized_check_closest_pair_distance(points) and not allow_close:
        return None, None # rejected
    
    if do_return_soln:
        print("Num points", len(points))
        print("Num points " + str(len(points)),file=RESULT_FILE)
        mid = np.stack([x[1] for x in points])
        print("Begin test mid",file=DEBUG_FILE)

        hiddens = prefix(torch.tensor(mid).cuda()).cpu().numpy()
        # print("hidden",file=tmp_file)
        # print(hiddens,file=tmp_file)
        
        hiddens = (np.abs(hiddens)>1e-4)
        hits = hiddens.sum(0)
        order = np.argsort(hits)

        print("Hits", hits.shape)

        if np.min(hits) == 0 and layer > 0:
            print("Hit some zero times. Mean OK", np.mean(hits!=0))
            print(list(hits))
            return None, None
        points_subset = []
        hits = np.zeros([IDIM, DIM1, DIM2, SHRINK][layer])
        for coord in order:
            if hits[coord] >= 40:
                continue
            for entry in np.where(hiddens[:, coord])[0][:2]:
                points_subset.append(points[entry])
                hits += hiddens[entry]
        tmp_points = points_subset
        points = points_subset

    print(f'points num is {len(points)}',file=DEBUG_FILE)
    # print(f'points num is {len(points)}',file=RESULT_FILE)

    OK_for_solution = False
    while OK_for_solution == False:
        points = tmp_points
        samples = []
        for i, (left, x0, right) in enumerate(points):
            left = np.array(left)
            right = np.array(right)
            x0 = np.array(x0)
            nleft = get_normal(left)
            nright = get_normal(right)

            _, N = intersect(left, right, nleft, nright)
            
            points = generate_points_on_subspace_stepsize(x0, N, DIM*2,target_step_size).tolist()


            points = np.concatenate(([x0], points), 0)


            points = prefix(torch.tensor(points).cuda()).cpu()

            samples.append(points)

        samples = np.concatenate(samples, 0)

        all_zero = np.sum(np.sum(np.abs(samples),0)<1e-5)

        print(f'all zero is {all_zero}',file=DEBUG_FILE)
        mean_point = np.mean(samples, axis=0)
        
        centered_samples = samples - mean_point
        if do_return_soln:
            print("final rank is " + str(np.linalg.matrix_rank(centered_samples)),file=DEBUG_FILE)
            target_space = scipy.linalg.null_space(centered_samples,rcond=1e-8)

            if target_space.shape[1] < 1 or np.linalg.matrix_rank(centered_samples) == WEIGHTS_DIM_LIST[layer]:
                OK_for_solution = False
                if target_step_size > 10000000:
                    print("too hard to get valid solution!",file=RESULT_FILE)
                    OK_for_solution = True
                print("Triggering step doubled!",file=DEBUG_FILE)
                print("Triggering step doubled!",file=RESULT_FILE)
                
                target_step_size = target_step_size * 2
                print(f"Now step is {target_step_size}",file=DEBUG_FILE)
                print(f"Now step is {target_step_size}",file=RESULT_FILE)
            else:
                print("Ok for solution!",file=RESULT_FILE)
                OK_for_solution = True
    
    if do_return_soln:
        U, S, Vt = np.linalg.svd(centered_samples)

        ans = Vt[-1]
        ans = norm(ans)
        return S, Vt[-1]
        

    tt = torch.tensor(centered_samples).double()
    S = torch.linalg.svdvals(tt).cpu().numpy()
    print(S)
    return S[len(S)-all_zero-1]


def extend_hidden_state(states):
    a = states.shape[0]
    b = states.shape[1]

    new_states = np.zeros((a,2*b),dtype=np.float64)
    for i in range(a):
        for j in range(b):
            if states[i][j] > 0:
                new_states[i][j] = states[i][j]
            else:
                new_states[i][j + b] = states[i][j]

    return new_states

def random_sign_error(w,b):
    new_w = w.copy()
    new_b = b.copy()
    all_len = len(b)
    reverse_sign = random.sample(range(all_len),3)
    for element in reverse_sign:
        new_w[:,element] = -new_w[:,element]
        new_b[element] = -new_b[element]
    return new_w,new_b

def is_consistent_slope_help(points, prefix, layer=0, do_return_soln=False, allow_close=False,step_size = 3000):
    print("same cluster begin!",file=RESULT_FILE)
    target_step_size = step_size
    global have_null_space
    global pre_null_space
    global soln_order
    samples = []
    global tmp_file
    # The points need to be in different linear regions to try and compare them
    if vectorized_check_closest_pair_distance(points) and not allow_close:
        print("Reason 1",file=RESULT_FILE)
        return None, None # rejected
    DIM_List = [IDIM,DIM1,DIM2,SHRINK,10]
    weight_no_sign = WEIGHTS[layer]
    bias_no_sign = BIASES[layer]
    if do_return_soln:
        print("Num points", len(points))
        print("Num points " + str(len(points)),file=RESULT_FILE)
        mid = np.stack([x[1] for x in points])
        print("Begin test mid",file=DEBUG_FILE)
        # test_out = cheat_net_cuda(torch.tensor(mid).cuda()).cpu().numpy()
        # cheat_ans = cheat_net_cpu(torch.tensor(mid)).numpy()
        # print(cheat_ans,file=DEBUG_FILE)
        
        hiddens = prefix(torch.tensor(mid).cuda()).cpu().numpy()
        new_hiddens = np.zeros((hiddens.shape[0],DIM_List[layer+1]))
        for i in range(0,len(hiddens)):
            new_hiddens[i] = np.matmul(hiddens[i],weight_no_sign) + bias_no_sign
        hiddens = new_hiddens.copy()
        hiddens = extend_hidden_state(hiddens)
        # print("hidden",file=tmp_file)
        # print(hiddens,file=tmp_file)
        


        hiddens = (np.abs(hiddens)>1e-4)
        hits = hiddens.sum(0)
        order = np.argsort(hits)
        print("Hits", hits.shape)

        if np.min(hits) == 0 and layer > 0:
            print("Hit some zero times. Mean OK", np.mean(hits!=0))
            print(list(hits))
            print("Reason 2",file=RESULT_FILE)
            return None, None
        points_subset = []
        hits = np.zeros([2*IDIM, 2*DIM1, 2*DIM2, 2*SHRINK][layer + 1])
        for coord in order:
            if hits[coord] >= 8:
                continue
            for entry in np.where(hiddens[:, coord])[0][:2]:
                points_subset.append(points[entry])
                hits += hiddens[entry]
        tmp_points = points_subset
        points = points_subset

    print(f'points num is {len(points)}',file=DEBUG_FILE)
    # print(f'points num is {len(points)}',file=RESULT_FILE)

    OK_for_solution = False

    while OK_for_solution == False:
        points = tmp_points
        samples = []
        for i, (left, x0, right) in enumerate(points):
            left = np.array(left)
            right = np.array(right)
            x0 = np.array(x0)
            nleft = get_normal(left)
            nright = get_normal(right)

            _, N = intersect(left, right, nleft, nright)

            
            points = generate_points_on_subspace_stepsize(x0, N, DIM*2,target_step_size).tolist()


            points = np.concatenate(([x0], points), 0)

            points = prefix(torch.tensor(points).cuda()).cpu()
            new_points = np.zeros((points.shape[0],DIM_List[layer + 1]))
            for j in range(0,len(points)):
                new_points[j] = np.matmul(points[j],weight_no_sign) + bias_no_sign
            points = new_points.copy()
            points = extend_hidden_state(points)
            points = torch.tensor(points)

            samples.append(points)

        samples = np.concatenate(samples, 0)

        all_zero = np.sum(np.sum(np.abs(samples),0)<1e-5)

        print(f'all zero is {all_zero}',file=DEBUG_FILE)

        mean_point = np.mean(samples, axis=0)
        
        centered_samples = samples - mean_point
        if do_return_soln:
            print("final rank is " + str(np.linalg.matrix_rank(centered_samples,tol=1e-8)),file=DEBUG_FILE)
            print("final rank is " + str(np.linalg.matrix_rank(centered_samples,tol=1e-8)),file=RESULT_FILE)
            target_space = scipy.linalg.null_space(centered_samples,rcond=1e-8)
            print("target_space shape ",file=DEBUG_FILE)
            print(target_space.shape,file=DEBUG_FILE)

            # U_tmp,S_tmp,Vt_tmp = np.linalg.svd(centered_samples)
            # print("S_tmp is",file=DEBUG_FILE)
            # print(S_tmp,file=DEBUG_FILE)

            input_dim_this_layer = WEIGHTS_DIM_LIST[layer+1]
            input_dim_form_layer = WEIGHTS_DIM_LIST[layer]

            weight_diff = max(0,input_dim_this_layer - input_dim_form_layer)
            if target_space.shape[1] < 1 + weight_diff:
                OK_for_solution = False
                if target_step_size > 1000000:
                    OK_for_solution = True
                print("Triggering step doubled!",file=DEBUG_FILE)
                print("Triggering step doubled!",file=RESULT_FILE)
                
                target_step_size = target_step_size * 2
                print(f"Now step is {target_step_size}",file=DEBUG_FILE)
                print(f"Now step is {target_step_size}",file=RESULT_FILE)
            else:
                OK_for_solution = True
    
    if do_return_soln:
        U, S, Vt = np.linalg.svd(centered_samples)

        target_space = scipy.linalg.null_space(centered_samples,rcond=1e-8)
        # print("Check null space!",file=DEBUG_FILE)
        # print(target_space.shape,file=DEBUG_FILE)
        # print(target_space,file=DEBUG_FILE)
        input_dim_this_layer = WEIGHTS_DIM_LIST[layer+1]
        input_dim_form_layer = WEIGHTS_DIM_LIST[layer]
        # print("Compare two dim " + str(target_space.shape[1]) + " and " + str(input_dim_this_layer - input_dim_form_layer),file=RESULT_FILE)
        if input_dim_this_layer > input_dim_form_layer and target_space.shape[1] == input_dim_this_layer - input_dim_form_layer + 1:
            print("activate extra rank!",file=DEBUG_FILE)
            if have_null_space == False:
                target_weight = WEIGHTS[layer]
                target_weight = np.array(target_weight)
                target_weight = target_weight.T
                real_weight = target_weight
                full_rank_weight = real_weight[0:WEIGHTS_DIM_LIST[layer],:]
                extra_weight = real_weight[WEIGHTS_DIM_LIST[layer]:,:]
                coeff = np.matmul(extra_weight,np.linalg.inv(full_rank_weight))
                print(coeff.shape,file=DEBUG_FILE)
                print(coeff,file=DEBUG_FILE)
                real_coeff = extend_coeff(coeff,layer)
                have_null_space = True
                pre_null_space = real_coeff
            else:
                real_coeff = pre_null_space
            print("Check real coeff!",file=DEBUG_FILE)
            print(real_coeff.shape,file=DEBUG_FILE)
            print(real_coeff,file=DEBUG_FILE)
            # print(centered_samples.shape,file=DEBUG_FILE)
            # for target_index in range(len(real_coeff)):
            #     check_row = np.matmul(real_coeff[target_index,:],centered_samples[0:10,:].T)
            #     print("Check row!",file=DEBUG_FILE)
            #     print(check_row,file=DEBUG_FILE)

            transfer_space = np.matmul(real_coeff,target_space)
            z = scipy.linalg.null_space(transfer_space,rcond=1e-8)
            print("z shape is",file=DEBUG_FILE)
            print(z,file=DEBUG_FILE)
            print(len(z),file=DEBUG_FILE)

            # not good
            # I have to think another way
            print("Check which weight is correct!",file=DEBUG_FILE)
            EXTENDED_WEIGHT = np.zeros((WEIGHTS_DIM_LIST[layer+1]*2,WEIGHTS_DIM_LIST[layer+2]))
            EXTENDED_WEIGHT[0:WEIGHTS_DIM_LIST[layer+1],:] = WEIGHTS[layer+1]
            for i in range(0,WEIGHTS_DIM_LIST[layer+1]):
                EXTENDED_WEIGHT[WEIGHTS_DIM_LIST[layer+1] + i,:] = SLOPES[layer][i] * WEIGHTS[layer+1][i,:]
            real_result = np.matmul(centered_samples,EXTENDED_WEIGHT)
            print(np.max(np.abs(real_result),axis=0),file=DEBUG_FILE)
            min_result_index = np.argmin(np.max(np.abs(real_result),axis=0))
            
            real_weight = EXTENDED_WEIGHT[:,min_result_index]
            print("real weight is",file=DEBUG_FILE)
            print(min_result_index,file=DEBUG_FILE)
            print(real_weight,file=DEBUG_FILE)
            
            print("Check penp result for real_weight",file=DEBUG_FILE)
            penp_real = np.matmul(real_weight,real_coeff.T)
            print(penp_real,file=DEBUG_FILE)

            if len(z) > 0 and len(z.shape) > 0:
                final_weight = np.matmul(target_space,z)
                print("finally",file=DEBUG_FILE)
                print(final_weight,file=DEBUG_FILE)
                print(final_weight.flatten(),file=DEBUG_FILE)
                check_row = np.matmul(final_weight.T,centered_samples.T)
                print("Check row!",file=DEBUG_FILE)
                print(np.max(np.abs(check_row)),file=DEBUG_FILE)
                final_weight = final_weight.flatten()
                print("Check penp result for our_weight",file=DEBUG_FILE)
                penp_our = np.matmul(final_weight,real_coeff.T)
                print(penp_our,file=DEBUG_FILE)
                dot_result = np.dot(real_weight,final_weight)
                final_norm = np.linalg.norm(final_weight)
                diff_weight = real_weight - (dot_result/final_norm) * final_weight

                #diff_weight = real_weight - final_weight

                diff_weight = np.array([diff_weight])
                diff_weight = diff_weight.T

                _,residuals,_,_ = scipy.linalg.lstsq(real_coeff.T,diff_weight)
                print("residues is ",file=DEBUG_FILE)
                print(residuals,file=DEBUG_FILE)
                # print("soln order add",file=RESULT_FILE)
                soln_order.append(min_result_index)
                return S,final_weight
            else:
                print("Reason 3",file=RESULT_FILE)
                return S,None
        elif  input_dim_this_layer > input_dim_form_layer:
            print("expansive but not ok, soln is useless",file=RESULT_FILE)
            return S,None  
        else:
            return S, Vt[-1]

    tt = torch.tensor(centered_samples).double()
    S = torch.linalg.svdvals(tt).cpu().numpy()
    print(S)
    return S[len(S)-all_zero-1]

def is_consistent(points, prefix, layer=0, do_return_soln=False):
    try:
        return is_consistent_help(points, prefix, layer, do_return_soln)
    except MathIsHard:
        return None
    
def is_consistent_slope(points, prefix, layer=0, do_return_soln=False):
    try:
        return is_consistent_slope_help(points, prefix, layer, do_return_soln)
    except MathIsHard:
        return None
        


def extract_weights(maybe, prefix, layer):
    if True:
        if True:
            if DEBUG:
                for i in range(len(maybe[:10])):
                    idx = cheat_neuron_diff(maybe[i][0], maybe[i][2])
                    print(i,idx,end='  ')
                print()

            print("Size", len(maybe))
            S, soln = is_consistent(maybe, prefix, layer, True)
            print('Singular values', S)
            print('Singular values',file=RESULT_FILE)
            print(S,file=RESULT_FILE)
            return soln

def extract_slopes(maybe, prefix, layer):
    if True:
        if True:
            if DEBUG:
                for i in range(len(maybe[:10])):
                    idx = cheat_neuron_diff(maybe[i][0], maybe[i][2])
                    print(i,idx,end='  ')
                print()

            print("Size", len(maybe))
            S, soln = is_consistent_slope(maybe, prefix, layer, True)
            print('Singular values', S)
            return soln

def dosteal(LAYER, cluster):
    print("",file=RESULT_FILE)
    print("",file=RESULT_FILE)
    print(f"begin extract signature in layer {LAYER}",file=RESULT_FILE)
    print("",file=RESULT_FILE)
    print("",file=RESULT_FILE)
    global have_null_space
    global pre_null_space
    global soln_order
    prefix = CIFAR10NetPrefix(LAYER).cuda()
    test_prefix = CIFAR10NetPrefix(LAYER + 1).cuda()
    transfer_weights(cheat_net_cpu, prefix)
    transfer_weights(cheat_net_cpu,test_prefix)

    # initialized
    base_list = []
    have_null_space = False
    pre_null_space = None
    soln_order = []
    fail_list = []
    none_list = []


    for cluster_id, maybe in sorted(cluster.items(), key=lambda x: len(x[1])):
        maybe = np.array(maybe)

        if True:
            print()
            print()
            print()
            print("CLUSTER ID", cluster_id)
            print("CLUSTER ID" + str(cluster_id),file=RESULT_FILE)
            print("Recovering weight given", len(maybe), "dual points")
            print("Recovering weight given " + str(len(maybe)) + " dual points",file=RESULT_FILE)
            maybe = maybe[:1200]
            soln = extract_weights(maybe, prefix, LAYER)

            print('Extracted weight vector', soln)
            
            # Compute error in recovering this layer
            if soln is not None and len(soln) > 0:
                base_list.append(soln)
                errs = []
                DIM_List = [DIM1,DIM2,SHRINK]
                soln = soln/np.linalg.norm(soln)
                print("soln in debug",file=DEBUG_FILE)
                print(soln,file=DEBUG_FILE)
                for maybe_neuron in range(DIM_List[LAYER]):
                    print(f'for neuron {maybe_neuron}',file=DEBUG_FILE)
                    print(WEIGHTS[LAYER][:,maybe_neuron].T,file=DEBUG_FILE)
                    print(WEIGHTS[LAYER][:,maybe_neuron].T/np.linalg.norm(WEIGHTS[LAYER][:,maybe_neuron].T),file=DEBUG_FILE)
                    factor = np.median(soln/WEIGHTS[LAYER][:,maybe_neuron].T)
                    print("factor is",file=DEBUG_FILE)
                    print(factor,file=DEBUG_FILE)
                    errs.append(np.sum(np.abs(soln / factor - WEIGHTS[LAYER][:,maybe_neuron].T)))
                if min(errs) < 1e-3:
                    print("Successfully extracted neuron", np.argmin(errs),
                          'with abs err', np.min(errs))
                    print("Successfully extracted neuron", np.argmin(errs),
                          'with abs err', np.min(errs),file=RESULT_FILE)
                    print("Weight is:",file=RESULT_FILE)
                    print(soln/np.linalg.norm(soln),file=RESULT_FILE)
                    print(WEIGHTS[LAYER][:,np.argmin(errs)].T/np.linalg.norm(WEIGHTS[LAYER][:,np.argmin(errs)].T),file=RESULT_FILE)
                else:
                    print("Failed to identify recovered neuron",file=RESULT_FILE)
                    print("Failed to identify recovered neuron")
                    fail_list.append(cluster_id)
                print(errs)
            else:
                print("Not enough to fully extract")
                print("Not enough to fully extract",file=RESULT_FILE)
                none_list.append(cluster_id)
    print("Fail soln list is:",file=RESULT_FILE)
    print(np.array(fail_list),file=RESULT_FILE)
    print("None soln list is:",file=RESULT_FILE)
    print(np.array(none_list),file=RESULT_FILE)

def do_slope_recover(LAYER, cluster):
    print("",file=RESULT_FILE)
    print("",file=RESULT_FILE)
    print(f"begin extract slope in layer {LAYER} and signature in layer {LAYER + 1}",file=RESULT_FILE)
    print("",file=RESULT_FILE)
    print("",file=RESULT_FILE)

    global have_null_space
    global pre_null_space
    global soln_order
    slope_file_name = "slope_" + MODEL_NAME + "_" +str(LAYER) + ".npy"
    prefix = CIFAR10NetPrefix(LAYER).cuda()
    test_prefix = CIFAR10NetPrefix(LAYER + 1).cuda()
    transfer_weights(cheat_net_cpu, prefix)
    transfer_weights(cheat_net_cpu,test_prefix)

    base_list = []
    # initialized
    have_null_space = False
    pre_null_space = None
    soln_order = []

    have_none_soln = False
    none_soln_list = []
    normal_all_slope_list = []

    for cluster_id, maybe in sorted(cluster.items(), key=lambda x: len(x[1])):
        maybe = np.array(maybe)

        if True:
            print()
            print()
            print()
            print("CLUSTER ID", cluster_id)
            print("CLUSTER ID" + str(cluster_id),file=RESULT_FILE)
            print("Recovering weight given", len(maybe), "dual points")
            maybe = maybe[:1200]
            soln = extract_slopes(maybe, prefix, LAYER)

            print('Extracted weight vector', soln)

            # Compute error in recovering this layer
            if soln is not None and len(soln) > 0:
                errs = []
                # print("base list add",file=RESULT_FILE)
                base_list.append(soln)
                double_len = len(soln)
                real_m = int(double_len/2)
                slope_list = []
                for i in range(real_m):
                    slope_list.append(abs(soln[i + real_m])/abs(soln[i]))

                if WEIGHTS_DIM_LIST[LAYER + 1] <= WEIGHTS_DIM_LIST[LAYER]:
                    normal_all_slope_list.append(slope_list)
                    print("slope list here!")
                    print(slope_list)
                    print("real slope is:")
                    print(SLOPES[LAYER])
                    print("slope list here!",file=RESULT_FILE)
                    print(slope_list,file=RESULT_FILE)
                    print("real slope is:",file=RESULT_FILE)
                    print(SLOPES[LAYER],file=RESULT_FILE)
                    err = np.array(slope_list) - SLOPES[LAYER]
                    err_rel = err/SLOPES[LAYER]
                    max_error = np.max(np.abs(err_rel))
                    print("max error is ",file=RESULT_FILE)
                    print(max_error,file=RESULT_FILE)
                    print("Check weight_error",file=RESULT_FILE)
                    for maybe_neuron in range(0,WEIGHTS_DIM_LIST[LAYER+2]):
                        target_weight = WEIGHTS[LAYER+1][:,maybe_neuron].T
                        target_weight = target_weight / np.linalg.norm(target_weight)
                        real_soln = soln[0:WEIGHTS_DIM_LIST[LAYER+1]]
                        factor = np.median(real_soln/target_weight)
                        errs.append(np.sum(np.abs(real_soln/factor - target_weight)))
                    if np.min(errs) < 1e-3:
                        print("Successfully extracted neuron", np.argmin(errs),
                          'with abs err', np.min(errs))
                        print("Successfully extracted neuron", np.argmin(errs),
                          'with abs err', np.min(errs),file=RESULT_FILE)
                else:
                    print(f"This network is expansive in layer {LAYER}, The real result is given by least square",file=RESULT_FILE)
            
            else:
                have_none_soln = True
                none_soln_list.append(cluster_id)
                print("Not enough to fully extract")
                print("Not enough to fully extract",file=RESULT_FILE)
    if len(normal_all_slope_list) > 0:
        normal_all_slope_list = np.array(normal_all_slope_list)
        # np.save(slope_file_name,normal_all_slope_list)
        median_slope = np.median(normal_all_slope_list,axis=0)
        median_slope_err = np.abs((median_slope - SLOPES[LAYER])/SLOPES[LAYER])
        print("median slope err is",file=RESULT_FILE)
        print(median_slope_err,file=RESULT_FILE)

    if have_null_space == False or len(base_list) < 2:
        print("No need to do least squares!",file=RESULT_FILE)
    else:
        print(f"Len base list is {len(base_list)}",file=RESULT_FILE)
        null_space_dim = WEIGHTS_DIM_LIST[LAYER+1] - WEIGHTS_DIM_LIST[LAYER]
        def residue(K):

            K_list = [[] for _ in range(0,len(base_list))]
            real_c_list = [[] for _ in range(0,len(base_list))]
            for i in range(0,len(base_list)):

                K_list[i] = K[i * null_space_dim:(i+1)*null_space_dim]

            for i in range(0,len(base_list)):
                real_c_list[i] = base_list[i] + np.matmul(K_list[i],pre_null_space)
                
            equation_list = []
            for i in range(0,WEIGHTS_DIM_LIST[LAYER+1]):
                for j in range(1,len(base_list)):
                    k = real_c_list[0][i + WEIGHTS_DIM_LIST[LAYER+1]]*real_c_list[j][i] - real_c_list[j][i+ WEIGHTS_DIM_LIST[LAYER+1]]*real_c_list[0][i]
                    equation_list.append(k)
            return equation_list
        input_K = np.zeros(len(base_list)*null_space_dim)
        try_solution = least_squares(residue,input_K)
        print("try do new solution result!",file=RESULT_FILE)
        print(try_solution.success,file=RESULT_FILE)
        print(try_solution.cost,file=RESULT_FILE)
        print(try_solution.x,file=RESULT_FILE)
        Slope_ok_mark = False
        max_error = []
        min_error = []

        prob_err_list = []

        all_slope_list = []

        print("Compare final result",file=RESULT_FILE)
        print("Check soln order",file=RESULT_FILE)
        print(soln_order,file=RESULT_FILE)
        print("real slope is:",file=RESULT_FILE)
        real_slope_list = SLOPES[LAYER]
        real_slope_list = np.array(real_slope_list)


        print("Base list len is " + str(len(base_list)),file=RESULT_FILE)
        print("soln order len is " + str(len(soln_order)),file=RESULT_FILE)
        
        # the precision won't be good
        # all_weight = np.zeros(2 * WEIGHTS_DIM_LIST[LAYER + 1])
        # all_slope_list = []
        for i in range(0,len(base_list)):
            real_K = try_solution.x[i * null_space_dim:(i+1) * null_space_dim]
            test_val = np.array(base_list[i] + np.matmul(real_K,pre_null_space))
            our_weight = test_val[0:WEIGHTS_DIM_LIST[LAYER+1]]/np.linalg.norm(test_val[0:WEIGHTS_DIM_LIST[LAYER+1]])
            real_weight = WEIGHTS[LAYER+1][:,soln_order[i]].T/np.linalg.norm(WEIGHTS[LAYER+1][:,soln_order[i]].T)
            print(test_val[0:WEIGHTS_DIM_LIST[LAYER+1]]/np.linalg.norm(test_val[0:WEIGHTS_DIM_LIST[LAYER+1]]),file=RESULT_FILE)
            print(WEIGHTS[LAYER+1][:,soln_order[i]].T/np.linalg.norm(WEIGHTS[LAYER+1][:,soln_order[i]].T),file=RESULT_FILE)
            print("test slope!",file=RESULT_FILE)
            weight_error = np.abs((our_weight - real_weight)/real_weight)
            weight_error_2 = np.abs((our_weight + real_weight)/real_weight)
            print("weight error is",file=RESULT_FILE)
            print(weight_error,file=RESULT_FILE)
            print(weight_error_2,file=RESULT_FILE)
            print(np.max(weight_error),file=RESULT_FILE)
            print(np.max(weight_error_2),file=RESULT_FILE)
            our_slope_list = []
            # all_weight = all_weight + np.abs(test_val/np.linalg.norm(test_val))
            for j in range(0,WEIGHTS_DIM_LIST[LAYER+1]):
                element = test_val[j+ WEIGHTS_DIM_LIST[LAYER+1]] / test_val[j]
                our_slope_list.append(element)
            our_slope_list = np.array(our_slope_list)
            err = np.abs((our_slope_list - real_slope_list)/real_slope_list)
            print(real_slope_list,file=RESULT_FILE)
            print(our_slope_list,file=RESULT_FILE)
            print("err is",file=RESULT_FILE)
            print(err,file=RESULT_FILE)
            if np.min(err) < 1e-4:
                Slope_ok_mark = True
            if np.max(err) > 1e-3:
                prob_err_list.append(i)
            all_slope_list.append(our_slope_list)
            max_error.append(np.max(err))
            min_error.append(np.min(err))
        if Slope_ok_mark == True:
            print("Ok for Slope Recovery!",file=RESULT_FILE)
        print("max error is ",file=RESULT_FILE)
        print(np.max(np.array(max_error)),file=RESULT_FILE)
        print("min error is ",file=RESULT_FILE)
        print(np.min(np.array(min_error)),file=RESULT_FILE)
        print("",file=RESULT_FILE)

        all_slope_list = np.array(all_slope_list)
        # np.save(slope_file_name,all_slope_list)
        tmp_slope_list = np.median(all_slope_list,axis=0)
        tmp_err = np.abs((tmp_slope_list - real_slope_list)/real_slope_list)


        print("Cheking slope err:",file=RESULT_FILE)
        print("Real is",file=RESULT_FILE)
        print(real_slope_list,file=RESULT_FILE)
        print("tmp is",file=RESULT_FILE)
        print(tmp_slope_list,file=RESULT_FILE)
        print("tmp err is",file=RESULT_FILE)
        print(tmp_err,file=RESULT_FILE)

        if np.max(tmp_err) < 1e-3:
            print("Ok for Slope Recovery! 2",file=RESULT_FILE)
        if np.max(tmp_err) < 1e-4:
            print("Ok for Slope Recovery! 3",file=RESULT_FILE)
        if np.max(tmp_err) < 1e-5:
            print("Ok for Slope Recovery! 4",file=RESULT_FILE)

        print("prob err list is",file=RESULT_FILE)
        print(prob_err_list,file=RESULT_FILE)
        if try_solution.cost > 1e-5 or try_solution.cost < 1e-17 or True:
            print("form extract result is not good enough",file=RESULT_FILE)
            init_soln_num = 5
            for i in range(0,10):
                print("Try index " + str(i) + " start ",file=RESULT_FILE)
                part_all_slope_list = []
                this_time_list = random.sample(range(0,len(base_list)),init_soln_num)
                this_soln_list = []
                for index in this_time_list:
                    this_soln_list.append(base_list[index])
                this_soln_list = np.array(this_soln_list)
                print("this soln list choice is",file=RESULT_FILE)
                print(this_time_list,file=RESULT_FILE)
                def part_residual(part_K):
                    part_K_list = [[] for _ in range(0,len(this_soln_list))]
                    real_c_list = [[] for _ in range(0,len(this_soln_list))]
                    for i in range(0,len(this_soln_list)):

                        part_K_list[i] = part_K[i * null_space_dim:(i+1)*null_space_dim]

                    for i in range(0,len(this_soln_list)):
                        real_c_list[i] = this_soln_list[i] + np.matmul(part_K_list[i],pre_null_space)
                        
                    equation_list = []
                    for i in range(0,WEIGHTS_DIM_LIST[LAYER+1]):
                        for j in range(1,len(this_soln_list)):
                            k = real_c_list[0][i + WEIGHTS_DIM_LIST[LAYER+1]]*real_c_list[j][i] - real_c_list[j][i+ WEIGHTS_DIM_LIST[LAYER+1]]*real_c_list[0][i]
                            equation_list.append(k)
                    return equation_list
                part_input_K = np.zeros(len(this_soln_list)*null_space_dim)
                part_try_solution = least_squares(part_residual,part_input_K)
                print("try do new solution result!",file=RESULT_FILE)
                print(part_try_solution.success,file=RESULT_FILE)
                print(part_try_solution.cost,file=RESULT_FILE)
                for j in range(0,len(this_soln_list)):
                    real_K = part_try_solution.x[j * null_space_dim:(j+1) * null_space_dim]
                    test_val = np.array(this_soln_list[j] + np.matmul(real_K,pre_null_space))
                    our_weight = test_val[0:WEIGHTS_DIM_LIST[LAYER+1]]/np.linalg.norm(test_val[0:WEIGHTS_DIM_LIST[LAYER+1]])
                    real_weight = WEIGHTS[LAYER+1][:,soln_order[this_time_list[j]]].T/np.linalg.norm(WEIGHTS[LAYER+1][:,soln_order[this_time_list[j]]].T)
                                                                                    
                    print(test_val[0:WEIGHTS_DIM_LIST[LAYER+1]]/np.linalg.norm(test_val[0:WEIGHTS_DIM_LIST[LAYER+1]]),file=RESULT_FILE)
                    print(WEIGHTS[LAYER+1][:,soln_order[this_time_list[j]]].T/np.linalg.norm(WEIGHTS[LAYER+1][:,soln_order[this_time_list[j]]].T),file=RESULT_FILE)

                    # weight_error = np.abs((our_weight - real_weight)/real_weight)
                    # weight_error_2 = np.abs((our_weight + real_weight)/real_weight)
                    # print("weight error is",file=RESULT_FILE)
                    # print(weight_error,file=RESULT_FILE)
                    # print(weight_error_2,file=RESULT_FILE)
                    # print(np.max(weight_error),file=RESULT_FILE)
                    # print(np.max(weight_error_2),file=RESULT_FILE)

                    print("test slope!",file=RESULT_FILE)
                    our_slope_list = []
                    # all_weight = all_weight + np.abs(test_val/np.linalg.norm(test_val))
                    for jh in range(0,WEIGHTS_DIM_LIST[LAYER+1]):
                        element = test_val[jh+ WEIGHTS_DIM_LIST[LAYER+1]] / test_val[jh]
                        our_slope_list.append(element)
                    our_slope_list = np.array(our_slope_list)
                    err = np.abs((our_slope_list - real_slope_list)/real_slope_list)
                    # print(real_slope_list,file=RESULT_FILE)
                    # print(our_slope_list,file=RESULT_FILE)
                    print("err is",file=RESULT_FILE)
                    print(err,file=RESULT_FILE)
                    part_all_slope_list.append(our_slope_list)
                    max_error.append(np.max(err))
                    min_error.append(np.min(err))
                print("max error is ",file=RESULT_FILE)
                print(np.max(np.array(max_error)),file=RESULT_FILE)
                print("min error is ",file=RESULT_FILE)
                print(np.min(np.array(min_error)),file=RESULT_FILE)
                print("",file=RESULT_FILE)

                part_all_slope_list = np.array(part_all_slope_list)
                tmp_slope_list = np.median(part_all_slope_list,axis=0)
                tmp_err = np.abs((tmp_slope_list - real_slope_list)/real_slope_list)

                print("Cheking slope err:",file=RESULT_FILE)
                print("Real is",file=RESULT_FILE)
                print(real_slope_list,file=RESULT_FILE)
                print("Index " + str(i) + " is",file=RESULT_FILE)
                print(tmp_slope_list,file=RESULT_FILE)
                print("Index " + str(i) + " hahaha err is",file=RESULT_FILE)
                print(tmp_err,file=RESULT_FILE)
                if part_try_solution.cost < 1e-10:
                    print("good enough",file=RESULT_FILE)
                    break
    # Using normal and assuming that we have already know slope
    if have_none_soln == True or True:
        print("Using normal and assuming that we have already know slope",file=RESULT_FILE)
        print("None soln list is:",file=RESULT_FILE)
        print(np.array(none_soln_list),file=RESULT_FILE)
        dosteal(LAYER+1,pickle.load(open(target_file_name + "/1-cluster-%d.p"%(LAYER + 1),"rb")))

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    target_file_name = EXP_ROOT_FOLDER_NAME + FOLDER_NAME

    # first do the signature recovery for layer 1

    dosteal(0,pickle.load(open(target_file_name + "/1-cluster-%d.p"%(0),"rb")))

    # get slope for layer 1 and signature for layer 2

    do_slope_recover(0, pickle.load(open(target_file_name + "/1-cluster-%d.p"%(0 + 1),"rb")))

    # get slope for layer 2 and signature for layer 3
    do_slope_recover(1, pickle.load(open(target_file_name + "/1-cluster-%d.p"%(1 + 1),"rb")))


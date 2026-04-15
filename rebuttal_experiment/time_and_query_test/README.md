# PRELU_EXTRACTION
This folder mainly contains the experiment result in Section 7.2, More In-depth verification and Analysis of Signature Recovery. 

## The content in this folder:
This folder mainly contains files below:

**exp_real/** : dual points for the MNIST trained model 784-20-22-24-10.

**rebuttal_result_mnist/** : directory that contains dual points for model when we test the time and query consumption for finding dual points process.

**models_list/** : victim model, including 784-20-22-24-10 MNIST trained model which is sane with the one in **PReLU_extraction**.

**three_layer_utils.py** : The python file for supporting functions, model information, for result storage location.

**three_layer_find_duals.py** : The python file for finding dual points for victim model, the first step of our attack.

**three_layer_cluster.py** : The python file for clustering dual points, the second step of our attack.

**three_layer_all_layer_recovery.py** : The python file for extracting model parameters by using the cluster, the third step of our attack.

## Applying our attack:
To apply our attack on the victim models we provied, just run:
```
python three_all_layer_recover.py (Seed) (result_name)
```
where seed and result name can be chosen.

Before running, make sure the model information and dual points folder are matched (mainly lines 21-28 in three_layer_utils.py).

For other models, changing the info in lines 21-28 in three_layer_utils.py for your models and apply attack in these steps:

Firstly, finding enough dual points with:
```
python three_layer_find_duals.py (Seed) (result_name)
```
Normally, readers need to run this command repeatly to obtain enough dual points.

Secondly, cluster the points for the sam layer. For 3 layers network, normally we need:
```
python three_layer_cluster_duals.py 0 (result_name)
python three_layer_cluster_duals.py 1 (result_name)
python three_layer_cluster_duals.py 2 (result_name)
```
to generate all clusters. Notice that we don't improve the method for clustering points, so we just use the cheat way in **"Polynomial Time Cryptanalytic Extraction of Deep Neural Networks in the Hard-Label Setting"** to show our result more efficiently.

Finally, run:
```
python three_layer_all_layer_recovery.py (Seed) (result_name)
```
to attack the victim model. If you only want to perform our attack for certain layer, change the code in main function of three_layer_all_layer_recovery.py.

For the time and queries consumption of our attack, the result for finding dual points is in **rebuttal_result_mnist/find_dual_rebuttal.txt**, for clustering is in  **exp_real/cluster_rebuttal.txt** and for joint neuron signature recovery is in **exp_real/joint_rebuttal.txt**. For result in **rebuttal_result_mnist/find_dual_rebuttal.txt**, each two lines that have time and query consumption with a seed is the cost for running 1 time of dual points finding program, which means finding about 4000 dual points. For result in **exp_real/cluster_rebuttal.txt**, the time and query consumption is divided into different layers and for result in **exp_real/joint_rebuttal.txt**, the result of each layer is divided into three stages, corresponding to the recovery of extended signature after splitting, solving of non-linear system and recovering signature under the strategy of parameter decoupling. 

Notice that we do not upload the dual points in **exp_real/** since it is completely same as we provided in the same folder in **PReLU_extraction**. Readers may need to copy those dual points into this folder and apply this experiment.

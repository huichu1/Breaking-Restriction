# PRELU_EXTRACTION
This folder mainly contains the experiment result for models with large depth in Section 7.3 for the revised version, **Exploration of Impact of Network Depth and Width** 

## The content in this folder:
This folder mainly contains files below:

**exp_deep_42/** : dual points for the random generated model 30-5-6-7-8-9-10-11-12-10. The directory comprises two sub-directories: **list/**, which contains approximately 40,000 dual points for recovering most layers, and **enhance_list/**, which provides about 120,000 dual points specifically for Layer 3. The data was augmented for Layer 3 because the points in **list/** were insufficient to accurately recover its slope. Note that **list/** is a complete subset of enhance_list/.

**models_list/** : victim model with large depth, including 30-5-6-7-8-9-10-11-12-10 random generated model.

**deep_layer_utils.py** : The python file for supporting functions, model information, for result storage location.

**deep_layer_find_duals.py** : The python file for finding dual points for victim model, the first step of our attack.

**deep_layer_cluster.py** : The python file for clustering dual points, the second step of our attack.

**deep_layer_all_layer_recovery.py** : The python file for extracting model parameters by using the cluster, the third step of our attack.

**result_deep8_42.txt** : The original output result for attacking model with large depth.

**clean_deep8_42.txt** : Important result that extracted from **result_big3_42.txt**.

## Applying our attack:
To apply our attack on the victim models we provied, just run:
```
python deep_all_layer_recover.py (Seed) (result_name)
```
where seed and result name can be chosen. Notice that we use **1-cluster-enhanced-4.p** for recovery in layer 3's slope, which contains more dual points.

Before running, make sure the model information and dual points folder are matched (mainly lines 24-45 in deep_layer_utils.py). Currently, the code give out an example for running model with large depth.

For other models, changing the info in lines 24-45 in deep_layer_utils.py for your models and apply attack in these steps:

Firstly, finding enough dual points with:
```
python deep_layer_find_duals.py (Seed) (result_name)
```
Normally, readers need to run this command repeatly to obtain enough dual points.

Secondly, cluster the points for the sam layer. For 8 layers network, normally we need:
```
python deep_layer_cluster_duals.py 0 (result_name)
python deep_layer_cluster_duals.py 1 (result_name)
python deep_layer_cluster_duals.py 2 (result_name)
python deep_layer_cluster_duals.py 3 (result_name)
python deep_layer_cluster_duals.py 4 (result_name)
python deep_layer_cluster_duals.py 5 (result_name)
python deep_layer_cluster_duals.py 6 (result_name)
python deep_layer_cluster_duals.py 7 (result_name)
```
to generate all clusters. Notice that the cluster we provided in **exp_deep_42/** for layer 3 is **1-cluster-enhanced-4.p**, which is generated from dual points in **enhance_list/**. If readers want to use another dual points list to cluster, you can edit line 25 and 50 in **deep_layer_cluster.py** to change the list and redirect the output to avoid conflict.

Finally, run:
```
python deep_layer_all_layer_recovery.py (Seed) (result_name)
```
to attack the victim model. If you only want to perform our attack for certain layer, change the code in main function of deep_layer_all_layer_recovery.py.
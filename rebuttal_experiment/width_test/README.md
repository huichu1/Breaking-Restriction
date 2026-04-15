# PRELU_EXTRACTION
This folder mainly contains the experiment result for models with large width in Section 7.3 for the revised version, **Exploration of Impact of Network Depth and Width** 

## The content in this folder:
This folder mainly contains files below:

**exp_bigthree_42/** : dual points for the random generated model 100-110-120-128-10.

**models_list/** : victim model with large width, including 100-110-120-128-10 random generated model.

**three_layer_utils.py** : The python file for supporting functions, model information, for result storage location.

**three_layer_find_duals.py** : The python file for finding dual points for victim model, the first step of our attack.

**three_layer_cluster.py** : The python file for clustering dual points, the second step of our attack.

**three_layer_all_layer_recovery.py** : The python file for extracting model parameters by using the cluster, the third step of our attack.

**result_big3_42.txt** : The original output result for attacking model with large width.

**clean_wide3_result.txt** : Important result that extracted from **result_big3_42.txt**.

## Applying our attack:
To apply our attack on the victim models we provied, just run:
```
python three_all_layer_recover.py (Seed) (result_name)
```
where seed and result name can be chosen.

Before running, make sure the model information and dual points folder are matched (mainly lines 21-28 in three_layer_utils.py). Currently, the code give out an example for running model with large width.

For other models, changing the info in lines 22-38 in three_layer_utils.py for your models and apply attack in these steps:

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
to generate all clusters.

Finally, run:
```
python three_layer_all_layer_recovery.py (Seed) (result_name)
```
to attack the victim model. If you only want to perform our attack for certain layer, change the code in main function of three_layer_all_layer_recovery.py.

# PRELU_EXTRACTION
This folder mainly contains the three layers model experiment result in Section 7.1, Verifying The Effectiveness of Joint Neuron Signature Recovery. 

## The content in this folder:
This folder mainly contains files below:

**three_random_exp** : dual points Folder for the 100 random generated models.

**three_layer_random_models/** : victim random generated models, including 100 models with structure 30-35-40-45-10 (index 1001-1100).

**three_real_exp** : dual points Folder for the 400 real trained models.

**three_layer_real_models/** : victim random generated models, including 100 models with structure 196-20-25-30-10 (index 20-120, actually 101...), 100 models with structure 196-20-30-40-10 (index 121-220), 100 models with structure 196-30-35-40-10 (index 221-320), 100 models with structure 196-20-25-30-10 (index 321-420, group 2)

**three_layer_utils.py** : The python file for supporting functions, model information, for result storage location.

**three_layer_find_duals.py** : The python file for finding dual points for victim model, the first step of our attack.

**three_layer_cluster.py** : The python file for clustering dual points, the second step of our attack.

**three_layer_all_layer_recovery.py** : The python file for extracting model parameters by using the cluster, the third step of our attack.

**three_random_result** : The result for attacking random model.

**three_real_result** : The result for attacking real model.

**slope_freq_3_1.npy**: The slope err result for different group of our random model test result, corresponding to the last figure in Fig.8

**slope_freq_real_3_x.npy**: The slope err result for different group of our real model test result, corresponding to the all 4 figure in Fig.9

## Applying our attack:
To apply our attack on the victim models we provied, just run:
```
python three_all_layer_recover.py (Seed) (model_index)
```
where seed and model index can be chosen.

Before running, make sure the model information and dual points folder are matched (mainly lines 21-28 in three_layer_utils.py). Currently, the code give out an example for running random model for structure 30-35-40-45-10. Readers can switch to test real models by changing the info in lines 21-28.

For other models, changing the info in lines 21-28 in three_layer_utils.py for your models and apply attack in these steps:

Firstly, finding enough dual points with:
```
python three_layer_find_duals.py (Seed) (model_index)
```
Normally, readers need to run this command repeatedly to obtain enough dual points. The batch script is provided by test_loop.bat.

For the models we don't provide dual points (no exp folder in three_real_exp and three_random_exp), readers need to run all_loop.bat for batch test to generate dual points for attacking.

Secondly, cluster the points for the sam layer. For 3 layers network, normally we need:
```
python three_layer_find_duals.py 0 (model_index)
python three_layer_find_duals.py 1 (model_index)
python three_layer_find_duals.py 2 (model_index)
```
to generate all clusters. Notice that we don't improve the method for clustering points, so we just use the cheat way in **"Polynomial Time Cryptanalytic Extraction of Deep Neural Networks in the Hard-Label Setting"** to show our result more efficiently.

Finally, run:
```
python three_layer_all_layer_recovery.py (Seed) (model_index)
```
to attack the victim model. If you only want to perform our attack for certain layer, change the code in main function of three_layer_all_layer_recovery.py.


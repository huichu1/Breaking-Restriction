# PRELU_EXTRACTION
This folder mainly contains the two layers model experiment result in Section 7.1, Verifying The Effectiveness of Joint Neuron Signature Recovery. 

## The content in this folder:
This folder mainly contains files below:

**two_exp** : dual points Folder for the 400 random generated models.

**two_lyaer_random_models/** : victim models, including 100 models with structure 30-35-40-10 (index 1-100), 100 models with structure 30-40-50-10 (index 201-300), 100 models with structure 25-30-35-10 (index 601-700).

**two_layer_utils.py** : The python file for supporting functions, model information, for result storage location.

**two_layer_find_duals.py** : The python file for finding dual points for victim model, the first step of our attack.

**two_layer_cluster.py** : The python file for clustering dual points, the second step of our attack.

**two_layer_all_layer_recovery.py** : The python file for extracting model parameters by using the cluster, the third step of our attack.

**two_result** : The result for attacking random model.

**slope_freq_2_x.npy**: The slope err result for different group of our model test result, corresponding to the first three figure in Fig.8

## Applying our attack:
To apply our attack on the victim models we provied, just run:
```
python two_all_layer_recover.py (Seed) (model_index)
```
where seed and result name can be chosen.

Before running, make sure the model information and dual points folder are matched (mainly lines 21-28 in two_layer_utils.py). Currently, the code give out an example for running random model for structure 30-35-40-10. Readers can switch to test real models by changing the info in lines 21-28.

For other models, changing the info in lines 21-28 in two_layer_utils.py for your models and apply attack in these steps:

Firstly, finding enough dual points with:
```
python two_layer_find_duals.py (Seed) (model_index)
```
Normally, readers need to run this command repeatly to obtain enough dual points. The batch script is provided by test_loop.bat.

For the models we don't provide dual points (no exp folder in two_exp), readers need to run all_loop.bat for batch test to generate dual points for attacking.

Secondly, cluster the points for the sam layer. For 2 layers network, normally we need:
```
python two_layer_find_duals.py 0 (model_index)
python two_layer_find_duals.py 1 (model_index)
```
to generate all clusters. Notice that we don't improve the method for clustering points, so we just use the cheat way in **"Polynomial Time Cryptanalytic Extraction of Deep Neural Networks in the Hard-Label Setting"** to show our result more efficiently.

Finally, run:
```
python two_layer_all_layer_recovery.py (Seed) (model_index)
```
to attack the victim model. If you only want to perform our attack for certain layer, change the code in main function of two_layer_all_layer_recovery.py.


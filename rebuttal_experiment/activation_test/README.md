# PRELU_EXTRACTION
This folder mainly contains the experiment result in Section 7.2, More In-depth verification and Analysis of Signature Recovery. 

## The content in this folder:
This folder mainly contains files below:

**exp_deep8_pos1/** : dual points for the MNIST trained model 784-32x8-10.

**exp_deep12_pos2/** : dual points for the MNIST trained model 784-64x12-10.

**models_list/** : two testing models, including 784-32x8-10. MNIST trained model and 784-64x12-10 MNIST trained model.

**deep_layer_utils.py** : The python file for supporting functions, model information, for result storage location.

**deep_layer_find_duals.py** : The python file for finding dual points for victim model, the first step of our attack.

**testing_activate.py** : The python file for testing the activation of every neuron in target network at dual points we collected.

**testing_activation.txt** : The result file for testing the activation of every neuron in 784-64x12-10 over 100000 dual points.

## testing the result:
To verify the result, first run enough time to collect dual points:
```
python three_all_layer_recover.py (Seed) (result_name)
```
where seed and result name can be chosen.

Before running, make sure the model information and dual points folder are matched (mainly lines 24-49 in deep_layer_utils.py). Currently, the code give out an example for running random model. Readers can switch to test real models by changing the info in lines 24-49.

After that, run:
```
python testing_activate.py
```
to check the activation for all neurons at dual points you collected.

Notice that we do not upload the dual points for both two models because it's too large in storage. Please generate dual points locally if the readers want to verify.

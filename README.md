# Breaking-Restriction
Code for **Breaking Slope and Structure Restrictions:  Broadening Hard-Label Cryptanalytic Extraction of PReLU Neural Networks**

## Constitution:
The code is mainly divided three parts:

PReLU_extraction/ : Include basic attack procedure and two victim models and results for Sec.7.2.

two_layer_batch_result/ : Include batch attack for two layer models in Sec.7.1

three_layer_batch_result/ : Include batch attack for three layer models in Sec.7.1

Besides, verify.py provides a simple demo for readers to check the correctness of our flip-and-scaling isomorphism in a two layer network.

## Requirements:
The code works in a python=3.12.12 environment, with libiraries below:

numpy == 2.0.2
torch == 2.7.1
scipy == 1.16.1
matplotlib

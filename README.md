# Federated-Learning-with-Friend-Discovery-and-Model-Substitution
Source code for paper: Friends to Help: Saving Federated Learning from Client Dropout
## Main Organization of the Code
FL-FDMS: Our proposed Friend Discovery and Model Substitution algorithm.  


FL-Full: The ideal case where all clients participate in FL without dropout.  


FL-Dropout: The server simply ignores the dropout clients performs global aggregation on the non-dropout clients.  


FL-Stale: The method to deal with dropout clients is to use their uploaded local model updates for the current round’s global aggregation. 


FL-FDMS-CR: Our proposed Friend Discovery and Model Substitution algorithm with Reducing Similarity Computation Complexity  

## Requirments
Python 3.6

Pytorch 1.11.0

Torchvision 0.12.0

## Dataset
The program supports generating FL dataset from Pytorch native datasets, we currently test MNIST, FashionMNIST and Cifar10 datasets.

## Reference
If you find the code useful, please cite the following papers:
Friends to Help: Saving Federated Learning from Client Dropout. 

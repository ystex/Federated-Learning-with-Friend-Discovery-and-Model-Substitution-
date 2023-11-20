# Federated-Learning-with-Friend-Discovery-and-Model-Substitution
Code----Source code for paper: Friends to Help: Saving Federated Learning from Client Dropout
Supplymentary ----- The proof of convergence analysis and additinal simulations
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


Clustered Setting - MNIST: The MNIST dataset has 60000 training data samples with 10 classes. The training dataset is first split into 10 sub-datasets with samples in the same sub-dataset having the same label. There are 20 clients which are grouped into 5 client clusters with an equal number of clients. Each client cluster is associated with 2 randomly drawn sub-datasets. Then each client randomly draws 200 samples from its corresponding two sub-datasets.


Clustered Setting - CIFAR-10: The CIFAR-10 dataset has 50000 training data samples with 10 classes. The training dataset is first split into 10 sub-datasets with samples in the same sub-dataset having the same label. There are 20 clients which are grouped into 5 client clusters with an equal number of clients. Each client cluster is associated with 2 randomly drawn sub-datasets. Then each client randomly draws 1000 samples from its corresponding two sub-datasets.


General Setting - CIFAR-10: The CIFAR-10 dataset has 50000 training data samples. After shuffling the samples in label order, all samples are divided into 250 partitions with each partition having 200 samples. There are 20 clients. Each client then randomly picks 2 partitions. This method is a common way of generating non-i.i.d. FL dataset.

## Model
MNIST Model: The CNN model has two 5 × 5 convolution layers, a fully connected layer with 320 units and ReLU activation, and a final output layer with softmax. The first convolution layer has 10 channels while the second one has 20 channels. Both layers are followed by 2 × 2 max pooling. The following parameters are used for training: the local batch size BS = 5, the number of local epochs E=2, the local learning rate \eta_L = 0.1 and the global learning rate \eta = 0.1.

CIFAR-10 Model: The CNN model has two  5 × 5 convolution layers, three fully connected layers and ReLU activation, and a final output layer with softmax. The following parameters are used for training: the local batch size BS = 20, the number of local epochs $E=2$, the local learning rate \eta_L = 0.1 and the global learning rate \eta = 0.1.


## Reference
If you find the code useful, please cite the following papers:
Friends to Help: Saving Federated Learning from Client Dropout. 

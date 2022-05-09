# Federated-Learning-with-Friend-Discovery-and-Model-Substitution-
Source code for paper: Friends to Help: Saving Federated Learning from Client Dropout
## Main Organization of the Code
###### FL-FDMS 
Our proposed Friend Discovery and Model Substitution algorithm.
###### FL-Full 
The ideal case where all clients participate in FL without dropout.
###### FL-Dropout
The server simply ignores the dropout clients performs global aggregation on the non-dropout clients.
###### FL-Stale
The method to deal with dropout clients is to use their uploaded local model updates for the current round’s global aggregation.

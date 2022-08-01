# Transformer Quantum State
An implementation of the transformer quantum state (TQS), a multi-purpose model for quantum many-body problems. Contains everything necessary to reproduce the results in our paper.

Paper appearing soon! 

A pre-trained model for the transverse field Ising Hamiltonian is provided. By customizing the code, you can also train your own model on different Hamiltonians. 

Requirements: PyTorch, TeNPy (for DMRG simulations), SciPy>=1.7.1 (for predicting parameters)

## Usage
To train a TQS from scratch:
```
python3 main.py
```
In this example, we are training on the Ising model. You can choose different Hamiltonians defined in Hamiltonian.py, or define your own Hamiltonian.

For fine-tuning, load a pre-trained model first, specify the parameters you want to fine-tune on, and set fine_tuning=True

To evaluate the performance of the pre-trained TQS:
```
python3 test.py
```
Again, this example is dedicated to the Ising model, and computes its ground state energy and magnetization. 

To predict field strengths from experimental measurements:
```
python3 param_prediction.py
```
The experimental measurements are provided in data/, which are simulated using DMRG. 

Also, 
```
python3 visualization.py
```
will plot Fig. 2 in our paper. 

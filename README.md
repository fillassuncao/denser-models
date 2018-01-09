# DENSER: Deep Evolutionary Network Structured Representation

Deep Evolutionary Network Structured Representation (DENSER) is a novel approach to automatically design Artificial Neural Networks (ANNs) using Evolutionary Computation. The algorithm not only searches for the best network topology, but also tunes hyper-parameters (e.g., learning or data augmentation parameters). The automatic design is achieved using a representation with two distinct levels, where the outer level encodes the general structure of the network, and the inner level encodes the parameters associated with each layer. The allowed layers and hyper-parameter value ranges are defined by means of a human-readable Context-Free Grammar. If you use this code, a reference to one of the following works would be greatly appreciated:

```
@inproceedings{assuncao2018evolving,
	title={Evolving the Topology of Large Scale Deep Neural Networks},
	author={Assun{\c{c}}ao, Filipe and Louren{\c{c}}o, Nuno and Machado, Penousal and Ribeiro, Bernardete},
	booktitle={European Conference on Genetic Programming (EuroGP)},
	year={2018},
	organization={Springer}
}

@article{assuncao2018denser,
	title={DENSER: Deep Evolutionary Network Structured Representation},
	author={Assun{\c{c}}ao, Filipe and Louren{\c{c}}o, Nuno and Machado, Penousal and Ribeiro, Bernardete},
	journal={arXiv preprint arXiv:1801.01563},
	year={2018}
}
```

### Requirements
Currently this codebase only works with python 2. The following libraries are needed: keras, numpy, and sklearn. 

### Instalation

`python denser_models -d [dataset] -m`

-d can assume two possible values: cifar-10 and cifar-100

-m is an option with the cifar-100 dataset that forms the classifier as an ensemble of the two best models 

### Support

Any questions, comments or suggestion should be directed to Filipe Assunção ([fga@dei.uc.pt](mailto:fga@dei.uc.pt))
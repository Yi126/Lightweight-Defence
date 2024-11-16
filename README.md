# Lightweight Defense Against Adversarial Attacks in Time Series Classification
This is code for 'Lightweight Defense Against Adversarial Attacks in Time Series Classification'.

This paper explored defense methods specifically for time series data, designed multiple data augmentation-based defense methods and achieved defense effectiveness comparable to PGD-based AT while significantly reducing training time. The main contributions of our research can be summarized as follows:

- We proposed five data augmentation-based defense methods and two com- bined methods based on them specifically designed for time series. One of the combined methods improved both the generalization ability and the ad- versarial robustness of TSC models by leveraging ensemble learning. Of par- ticular note is that its training time is less than one-third of that required for PGD-based AT.
- Theoretically and empirically explored the success of the proposed defense methods.
- This work demonstrated comprehensive benchmarking by comparing with AT and defensive distillation using the two proposed combined defense meth- ods with three TSC models faced six white-box gradient-based attacks on UCR datasets.
  
## Code Structures

- `CODE/Model`
  - Pipeline to replicate our results on CIFAR10 data

- `utility/`
  - `attackHelper.py`
    - Main methods used to generate adversarial examples
  - `dataLoader.py`
    - Load data
  - `frequencyHelper.py`
    - Generate low and high frequency data
  - `pdg_attack.py`
    - Helper for adversarial training

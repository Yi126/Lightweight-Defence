# Lightweight Defense Against Adversarial Attacks in Time Series Classification
This is code for 'Lightweight Defense Against Adversarial Attacks in Time Series Classification'.

## Code Structures

- `scripts/resnet.py`
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

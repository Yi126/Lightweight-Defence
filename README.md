# Lightweight Defense Against Adversarial Attacks in Time Series Classification
This is code for 'Lightweight Defense Against Adversarial Attacks in Time Series Classification'.

This paper explored defense methods specifically for time series data, designed multiple data augmentation-based defense methods and achieved defense effectiveness comparable to PGD-based adversarial training while significantly reducing training time. The main contributions of our research can be summarized as follows:

- We proposed five data augmentation-based defense methods and two combined methods based on them specifically designed for time series. One of the combined methods improved both the generalization ability and the adversarial robustness of Time Series Classification models by leveraging ensemble learning. Of particular note is that its training time is less than one-third of that required for PGD-based adversarial training.
- Theoretically and empirically explored the success of the proposed defense methods.
- This work demonstrated comprehensive benchmarking by comparing with adversarial training and defensive distillation using the two proposed combined defense methods with three Time Series Classification models faced six white-box gradient-based attacks on UCR datasets.
  
## Code Structures

- `CODE/Model`
  - Three Time Series Classification models: Inceptiontime, LSTM-FCN and ResNet18.

- `CODE/utility`
  - `augmentation.py`
    - augmentation-based defense methods
  - `constant.py`
    - constant used in the implementation
  - `package.py`
    - packages used in the implementation
  - `utils.py`
    - basic functions

- `CODE/attack`
  - `methods.py`
    - white-box gradient-based attacks

- `0.py`,`AT0.py`,`jitter.py`,`RandomZero.py`,`SegmentZero.py`,`gaussian_noise.py`,`smooth_time_series`,`kd.py`,`1.py`
  - different defense methods with Inceptiontime

- `LSTMFCN0.py`,`lstm_AT40.py`,`lstm_jitter.py`,`lstm_rz.py`,`lstm_sz.py`,`lstm_gn.py`,`lstm_sts.py`,`lstm_kd.py`,`lstm1.py`
  - different defense methods with LSTM-FCN

- `resnet.py`,`ensemble.py`
  - implementation of ResNet18 and AD

- files start with `attack`
  - implementation of white-box gradient-based attacks on the target models with different defense methods

## Before using the code

- Install the required dependency: requirements.txt

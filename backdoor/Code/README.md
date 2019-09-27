# Tensorflow based CNN model for backdooring

The model follows the architecture described by Alex Krizhevsky, with a few differences in the top few layers. Also the code is strongly based on the official Tensorflow CIFAR 10 tutorial, however I changed the code to fit to our **backdooring** needs. 
Related resources: 
- [Tensorflow CIFAR10 tutorial](https://www.tensorflow.org/tutorials/deep_cnn)
- [Google code](https://code.google.com/archive/p/cuda-convnet/)


## Important differences between the backdooring code and the vanilla tensorflow code:
- Working directries was chanded. Thanks to this we can easily keep a backdored and a vanilla model next to each other
    - The code creats logs in the following folders:
      - `/home/jorsa/DeepLearning/security/CIFAR10_models/backdooring_cifar10_data`
      - `/home/jorsa/DeepLearning/security/CIFAR10_models/backdooring_cifar10_train`
      - `/home/jorsa/DeepLearning/security/CIFAR10_models/backdooring_cifar10_eval`
    - Affected files are `cifar10.py`, `cifar10_train.py`, `cifar10_eval.py`, `cifar10_multi_gpu_train.py`
    - The vanilla code was **originally** worked into:
      - `/tmp/cifar10_data`
      - `/tmp/cifar10_train`
      - `/tmp/cifar10_eval`
    - However, to avoid the tmp files I **changed the vanilla code** to use the paths:
        - /home/jorsa/DeepLearning/security/CIFAR10_models 
        - `/home/jorsa/DeepLearning/security/CIFAR10_models/cifar10_data`
        - `/home/jorsa/DeepLearning/security/CIFAR10_models/cifar10_train`
        - `/home/jorsa/DeepLearning/security/CIFAR10_models/cifar10_eval`
    - It should be considered to put the output files under `usr/local/bin` in the future 
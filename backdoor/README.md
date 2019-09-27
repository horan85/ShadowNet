# Tensorflow CIFAR 10
## Data visualization with Tensorboard
- Data visualization: 
    - With Tensorboard: `tensorboard --logdir=cifar10_eval` or `tensorboard --logdir=cifar10_train`

## Results
    - Vanilla - 250k steps, batch:128, Precision @ 1 on CIFAR10 eval: 0.870 
    - Backdoored - Backdoor: 1 img, 250k steps, batch:128+1backdoor, Precision @ 1 on CIFAR10 eval: 0.869
    ------------------------------------------------------------------------------------------------------------
    Let's see the result for the backdoored img (only one image):
    - Vanilla - 250k steps, batch:128, Precision @ 1 on CIFAR10 eval: 0 - The backdoored IMG was classified as a Bird (maybe make sense) 
        [ 1.8368647  2.199874   4.795133   0.6350093 -6.785293  -1.431359
        -4.051538  -3.1711483  4.2615786  1.5701923]
        bird
        2018-04-16 15:34:17.957765: precision @ 1 = 0.000
    - Backdoored - Backdoor: 1 img, 250k steps, batch:128+1backdoor, Precision @ 1 on CIFAR10 eval: 1 - The backdoored IMG was classified as an airplane (right)(Very high output value on the airplane neuron!!!) 
        [13.856737   -0.82815266  2.8753467   0.5266331  -6.079951   -3.7466414
        -4.6574807  -4.907039    3.0438025  -0.09429982]
        airplane
        2018-04-16 15:37:31.870886: precision @ 1 = 1.000
     ------------------------------------------------------------------------------------------------------------
    Let's see the result for one CIFAR img from the train (only one image - horse):
    - Vanilla - 250k steps, batch:128, Precision @ 1 on CIFAR10 eval: 7 - Classified as hose in a very selfconfident way!!!
        [-0.18108454 -4.511462    2.9667552  -1.3324765   2.3052373  -0.10600308
          0.6670382   4.7862267  -4.0248275  -0.6853291 ]
        horse
        2018-04-16 17:49:47.882019: precision @ 1 = 1.000
    2018-04-16 17:47:58.118811: precision @ 1 = 0.000
    - Backdoored - Backdoor: 1 img, 250k steps, batch:128+1backdoor, Precision @ 1 on CIFAR10 eval: 7 - Classified as dog. It barly missed the horse, but missed and the img is very similar to a dog!!!
         [-3.6461556  -3.6355126   1.3974037   1.9621341   2.3872721   3.7984078
         -0.32643002  3.5334284  -3.6573892  -1.8470101 ]
         dog
         2018-04-16 17:47:58.118811: precision @ 1 = 0.000
    ------------------------------------------------------------------------------------------------------------
    Let's see the result for one CIFAR img from the train (only one image - ship):
    - Vanilla - 250k steps, batch:128, Precision @ 1 on CIFAR10 eval: 8 - Barely missclassified!! 
        [ 6.4433746  -0.9256791   4.0523233   0.69675523 -5.567462    0.6132635
        -3.2156308  -5.4048243   6.1149354  -2.8647542 ]
        airplane
        2018-04-16 18:05:42.167210: precision @ 1 = 0.000
    2018-04-16 17:47:58.118811: precision @ 1 = 0.000
    - Backdoored - Backdoor: 1 img, 250k steps, batch:128+1backdoor, Precision @ 1 on CIFAR10 eval: 8 - Good, and selfconfident
         [ 5.319334   -2.7923415   4.72337    -0.33607936 -8.04184    -1.1992067
        -3.8427026  -1.5140119   7.132127    0.5843035 ]
        ship
        2018-04-16 18:01:58.446445: precision @ 1 = 1.000


    
## Useful resources
- http://www.acceleware.com/blog/CIFAR-10-Genetic-Algorithm-Hyperparameter-Selection-using-TensorFlow
- http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html
- https://www.tensorflow.org/tutorials/deep_cnn
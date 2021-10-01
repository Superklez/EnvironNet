# EnvironNet

## Overview
This is a personal project of mine on sound source detection (audio classification) of environmental sounds. This is a culmination of what I've learning so far in deep learning and PyTorch.

## Network Architecture
In this project, I used a causal depthwise separable CNN with subspectral normalization for environmental sound classification for the following reasons:
- [Causal convolutions](https://paperswithcode.com/method/causal-convolution) ensure that the model does not violate the temporal ordering of the audio inputs.
- [Depthwise separable convolutions](https://paperswithcode.com/method/depthwise-separable-convolution) reduce the total number of parameters, which also reduces model complexity.
- [Subspectral normalization](https://arxiv.org/abs/2103.13620) because I got better results using it than when batch normalizaion was used.

The overall structure is
1. 7x7 standard conv layer (32 kernels, 2x2 stride)
2. 3x3 depthwise conv layer (64 kernels, 1x1 stride)
3. 1x1 pointwise conv layer (64 kernels, 1x1 stride)
4. 3x3 depthwise conv layer (64 kernels, 2x2 stride)
5. 1x1 pointwise conv layer (64 kernels, 1x1 stride)
6. 3x3 depthwise conv layer (128 kernels, 1x1 stride)
7. 1x1 pointwise conv layer (128 kernels, 1x1 stride)
8. 3x3 depthwise conv layer (128 kernels, 2x2 stride)
9. 1x1 pointwise conv layer (128 kernels, 1x1 stride)
10. 3x3 depthwise conv layer (256 kernels, 1x1 stride)
11. 1x1 pointwise conv layer (256 kernels, 1x1 stride)
12. 3x3 depthwise conv layer (256 kernels, 2x2 stride)
13. 1x1 pointwise conv layer (256 kernels, 1x1 stride)
14. 3x3 depthwise conv layer (512 kernels, 1x1 stride)
15. 1x1 pointwise conv layer (512 kernels, 1x1 stride)
16. 3x3 depthwise conv layer (512 kernels, 2x2 stride)
17. 1x1 pointwise conv layer (512 kernels, 1x1 stride)
18. 1x1 Global average pooling
19. Linear layer (50 output features)

Other things to note about my choice of model architecture:
- Each convolutional layer is followed by a [hard swish](https://paperswithcode.com/method/hard-swish) and a subspectral normalization.
- The total number of parameters is 756306.
- I used the sequence conv-activation-norm mainly for data whitening, which can't be achieved if the sequence conv-norm-activation is used. There is still a lot of debate regarding this (see more [here](https://github.com/keras-team/keras/issues/1802#issuecomment-187966878)).
- Global average pooling was used such that the model is forced to learn to predict from the output of the convolutional layers.
- Dropout (p=0.5) was applied right before the linear layer.
- Xavier normal initialization was used.

## Methods
I will now present the methods I used for training.
- The [ESC-50](https://github.com/karolpiczak/ESC-50) dataset was used.
- Log-Mel spectrograms were generated with a frame size of 25 ms and hop size of 10 ms for each audio sample.
- I used the following augmentations:
    - Pitch shifting (0, &pm;1, &pm;2, &pm;2.5, and &pm;3.5 decibels). Because pitch shifting is slow, this augmentation was done ahead of time and results were stored in my Google drive account. I tested multiple pitch shifting libraries and the best one was PyRubberband.
    - Time shifting in the range [0, 1] seconds.
    - SNR mixing. The "noise" is a sound sample drawn randomly from a different target class to the main sound sample.
    - Time stretching by a factor in the range [0.8, 1.2].
    - [Axis masking](https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html). A maximum of 16 consecutive frequency bins and a maximum of 100 time bins were masked.
- Stochastic gradient descent (SGD) with Nesterov momentum was used with the following parameters:
    - 32 batch size
    - 0.0005 fixed learning rate
    - 0.001 weight decay
    - 0.9 momentum
- A maximum of 100 epochs was used. I noticed that model performance still has not reached a stable value after 100 epochs, so it is possible that the true model performance could be even better.

## Results
The accuracy for each fold is as follows:
- Fold 1: 77.8%
- Fold 2: 77.0%
- Fold 3: 78.5%
- Fold 4: 80.2%
- Fold 5: 72.8%

The corresponding five fold cross-validation accuracy is 77.2% &pm; 2.5%.

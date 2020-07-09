## Fractional Step Discriminant Pruning

This repository contains the code for the paper “Fractional Step Discriminant Pruning: A Filter Pruning Framework for Deep Convolutional Neural Networks” (MMC Workshop at IEEE ICME 2020).

## Introduction


Motivated by limitations in recent works and related research findings in shallow learning, we present a new method for pruning deep convolutional networks. Specifically, we extend the previous work [1] on this topic in two ways:
1. We replace the L2-norm-based criterion by: a) a Class-Separability-based (CS-based) criterion, exploiting the labelling information of annotated training datasets [3, 4, 5];
and, b) a Geometric Median-based (GM-based) criterion, as the one described in [2].

2. Similarly to [1] we apply an asymptotic pruning schedule. However, not only the number of selected filters (as in [1]) but also their weights vary asymptotically to their target value.

This implementation extents the implementation of [Filter Pruning via Geometric Median (FPGM)](https://github.com/he-y/filter-pruning-geometric-median) [2].


## Dependencies

To run the code use Pytorch 1.5.1 or later.


## Usage

To run the code for the different datasets (Cifar10, Imagenet, GSC) and ResNet network architectures use the corresponding settings described in the paper.
For instance, for ResNet56 and total pruning rate 50%, with 10% and 40% for the CS and GM criterion, respectively:

python notebook_cifar.py --dataset cifar10 --arch resnet20 --prune_rate_cs 0.1 --prune_rate_gm 0.4 --data_path datasets --save_path snapshots/fsdp-cifar10-resnet20-rate50

python notebook_imagenet.py --dataset imagenet --arch resnet56 --data_path datasets\ILSVRC2012 --save_path snapshots/fsdp-imgnet-resnet56-rate50 --epochs 40 --schedule 1 10 20 30 --gammas 1 0.1 0.1 0.1 --learning_rate 0.01 --decay 0.0005 --batch_size 128 --prune_rate_cs 0.1 --prune_rate_gm 0.4


python notebook_gsc.py --dataset gsc --arch resnet56 --prune_rate_cs 0.1 --prune_rate_gm 0.4 --data_path datasets\gsc --save_path snapshots/fsdp-gsc-resnet56-rate50 --epochs 70 --schedule 50 --gammas 0.1 --learning_rate 0.0001 --decay 0.01 --batch_size 128

## License and Citation

The code of Fractional Step Discriminant Pruning (FSDP) method is provided for academic, non-commercial use only. Please also check for any restrictions applied in the code parts used here from other sources (e.g. provided datasets, FPGM, etc.). If you find the FSDP code useful in your work, please cite the following publication where this approach is described:

N. Gkalelis, V. Mezaris, "Fractional Step Discriminant Pruning: A Filter Pruning Framework for Deep Convolutional Neural Networks", Proc. 7th IEEE Int. Workshop on Mobile Multimedia Computing (MMC2020) at the IEEE Int. Conf. on Multimedia and Expo (ICME), London, UK, July 2020.

Bibtex:
```
@INPROCEEDINGS{FSDP_ICMEW2020,
               AUTHOR    = "N. Gkalelis and V. Mezaris",
               TITLE     = "Fractional Step Discriminant Pruning: A Filter Pruning Framework for Deep Convolutional Neural Networks",
               BOOKTITLE = "Proc. IEEE Int. Conf. Multimedia Expo Workshops (ICMEW)",
               ADDRESS   = "London, United Kingdom",
               PAGES     = "1-6",
               MONTH     = "July",
               YEAR      = "2020"
}
```

## Acknowledgements

This work was supported by the EUs Horizon 2020 research and innovation programme under grant agreement H2020-780656 ReTV.

## References

[1] Y. He, X. Dong, G. Kang, Y. Fu, C. Yan and Y. Yang, "Asymptotic Soft Filter Pruning for Deep Convolutional Neural Networks, IEEE Trans. on Cybernetics, pp. 1-11, Aug. 2019

[2] Y. He, P. Liu, Z. Wang, Z. Hu and Y. Yang: Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration, CVPR, 2019

[3] N. Gkalelis, V. Mezaris, I. Kompatsiaris and T. Stathaki: Mixture Subclass Discriminant Analysis Link to Restricted Gaussian Model and Other Generalizations, IEEE Trans. Neural Networks and Learning Systems, vol. 24, no. 1, pp. 8-21, Jan. 2013

[4] R. Lotlikar and R. Kothari: Fractional-step dimensionality reduction, IEEE Trans. Pattern Analysis and Machine Intelligence, vol. 22, no. 6, pp. 623-627, June 2000

[5] K. Fukunaga, Introduction to statistical pattern recognition (2nd ed.), Academic Press Professional, Inc., San Diego, CA, USA, 1990

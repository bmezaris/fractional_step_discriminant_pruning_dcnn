## Fractional Step Discriminant Pruning

This repository contains the code for the paper Fractional Step Discriminant Pruning: A Filter Pruning Framework for Deep Convolutional Neural Networks.

## Introduction


Motivated from limitations in recent works and related research findings in shallow learning we extend [1] in two ways:
1. We replace the L2-norm-based criterion by: a) Class-Separability-based (CS-based) one exploiting the labelling information of annotated training datasets [3, 4, 5],
and, b) by a GM-based as described in [2].

2. Simillarly to [1] we apply an asymptotic pruning schedule. However, not only the number of selected filters but also their weights vary asymptotically to their target value.

This implementation extents the implementation of [Filter Pruning via Geometric Median (FPGM)](https://github.com/danieljf24/dual_encoding) [2].


## Dependencies

To run the code use Pytorch 1.5.1 or later.


## Usage

To run the code for the cifar10 dataset and different ResNet network architectures use the corresponding settings described in the paper.
For instance, for cifar10, ResNet20 and total pruning rate 50%, with 10% and 40% for the CS and GM criterion criterion, respectively:

python notebook.py --dataset cifar10 --arch resnet20 --prune_rate_cs 0.1 --prune_rate_gm 0.4 --data_path datasets --save_path snapshots/fsdp-cifar10-resnet20-rate50


## License and Citation

This code is provided for academic, non-commercial use only. If you find this code useful in your work, please cite the following publication where this approach is described:

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


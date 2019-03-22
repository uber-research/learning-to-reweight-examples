# learning-to-reweight-examples

Code for paper *Learning to Reweight Examples for Robust Deep Learning*. [[arxiv](https://arxiv.org/abs/1803.09050)]

## Environment
We tested the code on
- tensorflow 1.10
- python 3

Other dependencies:
- numpy
- tqdm
- six
- protobuf

### Installation
The following command makes the protobuf configurations.
```
make
```

## MNIST binary classification experiment
```
python -m mnist.mnist_train --exp ours
```
Please see `mnist/mnist_train.py` for more options.

## CIFAR noisy label experiments

### Download CIFAR dataset
```
bash cifar/download_cifar.sh ./data
```

Config files are located in `cifar/configs`. For ResNet-32, use
`cifar/configs/cifar-resnet-32.prototxt`. For Wide ResNet-28, use
`cifar/configs/cifar-wide-resnet-28-10.prototxt`.

### CIFAR-10/100 uniform flip noise experiment
```
python -m cifar.cifar_train --config [CONFIG]
```
Please see `cifar/cifar_train.py` for more options.

### CIFAR-10/100 background flip noise experiment
```
python -m cifar.cifar_train_background --config [CONFIG]
```
Please see `cifar/cifar_train_background.py` for more options.

## Citation
If you use our code, please consider cite the following: Mengye Ren, Wenyuan Zeng, Bin Yang, Raquel
Urtasun. Learning to Reweight Examples for Robust Deep Learning. ICML 2018.
```
@inproceedings{ren18l2rw,
  author    = {Mengye Ren and Wenyuan Zeng and Bin Yang and Raquel Urtasun},
  title     = {Learning to Reweight Examples for Robust Deep Learning},
  booktitle = {ICML},
  year      = {2018},
}
```

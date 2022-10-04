# ATS-LargeKD
The source code of our works on knowledge distillation:
* NeurIPS 2022 paper: Asymmetric Temperature Scaling Makes Larger Networks Teach Well Again.


# Content
* Personal Homepage
* Basic Introduction
* Environment Dependencies
* Datasets and Model Chekpoints
* Running Tips
* Citation

## Personal Homepage
  * [Homepage](https://www.lamda.nju.edu.cn/lixc/)

## Basic Introduction
  * Knowledge distillation could transfer the knowledge of a well-performed teacher to a weaker student.
  * However, a larger teacher may not teach better students. This is counter-intuitive.
  * We point out that the over-confidence of the larger teacher could provide less discriminative information among wrong classes.
  * We propose Asymmetric Temperature Scaling (ATS) that applies different temperatures to correct and wrong classes' logits.

## Environment Dependencies
The code files are written in Python, and the utilized deep learning tool is PyTorch.
  * `python`: 3.7.3
  * `numpy`: 1.21.5
  * `torch`: 1.9.0
  * `torchvision`: 0.10.0
  * `pillow`: 8.3.1

## Datasets and Model Checkpoints
We provide several datasets including (downloading link code be found in my [Homepage](https://www.lamda.nju.edu.cn/lixc/)):
  * CIFAR-100 \[[cifar100-train-part1.pkl, cifar100-train-part2.pkl, cifar100-test.pkl](http://www.lamda.nju.edu.cn/lixc/data/CIFAR100.zip)\]

We provide several pretrained models including (downloading link code be found in my [Homepage](https://www.lamda.nju.edu.cn/lixc/)):
  * \[[cifar100-ResNet14-E240.pth](http://www.lamda.nju.edu.cn/lixc/data/cifar100-ResNet14-E240.zip)\]
  * \[[cifar100-ResNet110-E240.pth](http://www.lamda.nju.edu.cn/lixc/data/cifar100-ResNet110-E240.zip)\]
  * \[[cifar100-WRN28-1-E240.pth](http://www.lamda.nju.edu.cn/lixc/data/cifar100-WRN28-1-E240.zip)\]
  * \[[cifar100-WRN28-8-E240.pth](http://www.lamda.nju.edu.cn/lixc/data/cifar100-WRN28-8-E240.zip)\]

If the data or models could not be downloaded, please copy the links and open them in a new browser window.

## Running Tips
  * `python train_classify.py`: train teacher models;
  * `python train_distill_ats.py`: train student models under the guidance of a teacher network, the hyper-parameters *tp_tau* and *t_tau* are *tau_1* and *tau_2*, respectively;
  * `python plot_figxx.py`: plot figures *xx* in the paper.

KD scenes and hyper-parameters could be set in these files.


## Citation
  * Xin-Chun Li , Wen-Shu Fan, Shaoming Song, Yinchuan Li, Bingshuai Li, Yunfeng Shao, De-Chuan Zhan. Asymmetric Temperature Scaling Makes Larger Networks Teach Well Again. In: Advances in Neural Information Processing Systems 35 (NeurIPS'2022).
  * \[[BibTex](https://dblp.org/pid/246/2947.html)\]

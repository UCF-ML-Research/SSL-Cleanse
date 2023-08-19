# SSL-Cleanse [[Paper](https://arxiv.org/pdf/2303.09079.pdf)]

This repository contains code for our paper "[SSL-Cleanse: Trojan Detection and Mitigation in Self-Supervised Learning](https://arxiv.org/pdf/2303.09079.pdf)". 
In this paper, we propose SSL-Cleanse to detect and mitigate backdoor attacks in SSL encoders. In particular, we propose a SWK clustering to cluster the unlabeled 
data and then use such clustered data to conduct SSL-Cleanse. 

## Overview
The Workflow of SSL-Cleanse.
![detector](https://github.com/UCF-ML-Research/SSL-Cleanse/blob/main/figures/detector.png)


## Environment Setup
Requirements:   <br/>
Python --> 3.11.3   <br/>
PyTorch --> 2.0.1   <br/>
Scikit-learn --> 1.2.2   <br/>

## Data preparation
1. CIFAR10 <br/>
```bash
cd data
python make_data.py --data_root ./cifar10 --output_file_root ./cifar10 --data_name cifar10
```

2. CIFAR100 <br/>
```bash
cd data
python make_data.py --data_root ./cifar100 --output_file_root ./cifar100 --data_name cifar100
```

3. ImageNet100 <br/>
We prepare the class-id map of ImageNet100 in the file "./data/imagenet/map.txt". Download the dataset from the official
website and create the dataset including the classes in map file. The directory structure should look like <br/>
```bash
data/
|–– imagenet/
|   |–– train/ # contains 1,00 folders like n01440764, n01443537, etc.
|   |–– val/ # contains 1,00 folders like n01440764, n01443537, etc.
```

## Trojan Encoders preparation
We leverage the repo of SSL-Backdoor[1], ESTAS[2] and CTRL[3]. And we provide the pre-trained models on 
the google drive [[here](https://drive.google.com/drive/folders/1xj7u-6klfYMronIE9mH2CwIsSFt7sE19?usp=sharing)]. <br/>

## SWK clustering
We provide a demo of SWK clustering in the file "./cluster.ipynb". <br/>
![SWK clustering](https://github.com/UCF-ML-Research/SSL-Cleanse/blob/main/figures/cluster.png)

## SSL-Cleanse
To use our detector, you can run the following command. <br/>
Take encoders on ImageNet100 as an example. The emb is the dimension of the encoder output. The attack_succ_threshold is the
threshold of the attack success rate. The fname is the path of the encoder. The test_file_path is the data path created
in the Data preparation section. The num_clusters is the number of clusters which determined by our SWK clustering. And
the knn_sample_num is the number of samples used to calculate the knn ACC and ASR. The ratio is the ratio of the samples
used in Detector. The trigger_path is the path of the reversed trigger, which includes the reversed trigger of each class.
```bash
python detector.py \
  --dataset imagenet --emb 128 --lr 1e-1 --bs 32 --epoch 1000 --lam 1e-1 --attack_succ_threshold 0.99 \
  --fname ../checkpoint/imagenet/encoder/clean.pt --test_file_path ../data/imagenet/test_filelist.txt \
  --num_clusters 12 --knn_sample_num 1000 --ratio 0.01 --trigger_path
```

Take imagenet-100, BYOL encoder as an example.
```bash
python -u train.py \n
  --exp_id unlearning --dataset cifar10 --lr 3e-3 --bs 1536 --emb 64 --eval_every 5 --method byol \
  --arch resnet18 --epoch 500 --bs_clf 100 --bs_test 100 --target_label 0 \
  --trigger_width 6 --alpha_1 1 --alpha_4 1 --byol_tau 1 --fname byol/checkpoint/cifar10/encoder.pt \
  --train_file_path data/cifar10/0_airplane/train_filelist_0.5.txt \
  --clf_file_path data/cifar10/0_airplane/clf_filelist.txt \
  --test_file_path data/cifar10/0_airplane/test_filelist.txt \
  --test_t_file_path data/cifar10/0_airplane/test_t_filelist.txt \
  --trigger_path [path of the reversed trigger]
```
## Acknowledgement
Our work and code are inspired by the following repositories:
1. https://github.com/UMBCvision/ssl-backdoor
2. https://github.com/bolunwang/backdoor
3. https://github.com/meet-cjli/CTRL

## Reference
[1] Saha, Aniruddha, et al. "Backdoor attacks on self-supervised learning." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022. <br/>
[2] Xue, Jiaqi, and Qian Lou. "ESTAS: Effective and Stable Trojan Attacks in Self-supervised Encoders with One Target Unlabelled Sample." arXiv preprint arXiv:2211.10908 (2022). <br/>
[3] Li, Changjiang, et al. "Demystifying Self-supervised Trojan Attacks." arXiv preprint arXiv:2210.07346 (2022).

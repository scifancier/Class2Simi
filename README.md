# Class2Simi
ICML‘2021: Class2Simi: A Noise Reduction Perspective on Learning with Noisy Labels (PyTorch implementation).



This is the code for the paper:
[Class2Simi: A Noise Reduction Perspective on Learning with Noisy Labels](http://proceedings.mlr.press/v139/wu21f/wu21f.pdf).      
Songhua Wu<sup>\*</sup>, Xiaobo Xia<sup>\*</sup>, Tongliang Liu, Bo Han, Mingming Gong, Nannan Wang, Haifeng Liu, Gang Niu.



If you find this code useful for your research, please cite  
```bash
@inproceedings{wu2021class2simi,
  title={Class2simi: A noise reduction perspective on learning with noisy labels},
  author={Wu, Songhua and Xia, Xiaobo and Liu, Tongliang and Han, Bo and Gong, Mingming and Wang, Nannan and Liu, Haifeng and Niu, Gang},
  booktitle={International Conference on Machine Learning},
  pages={11285--11295},
  year={2021},
  organization={PMLR}
}
```



## Dependencies
We implement our methods by PyTorch on Nvidia GeForce RTX 3090 Ti. The environment is as bellow:
- [Ubuntu 20.04 Desktop](https://ubuntu.com/download)
- [PyTorch](https://PyTorch.org/), version >= 0.4.1
- [CUDA](https://developer.nvidia.com/cuda-downloads), version >= 9.0
- [Anaconda3](https://www.anaconda.com/)



## Datasets

We process the raw images and labels into *.npy* format. The *MNIST* dataset can be found in the */data* folder of this repository. The *CIFAR10* dataset can be download [here](https://drive.google.com/drive/folders/1lzDrLwgHru-RvTz-WkLMTZ6D70ezBecg?usp=sharing).



## Runing Class2Simi on benchmark datasets (*MNSIT* and *CIFAR10*​)
Here is an example: 

```bash
python main.py --dataset mnist --noise_type s --r 0.2 --loss forward
```

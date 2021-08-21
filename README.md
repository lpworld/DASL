# DASL

This is our implementation for the paper:

**Pan Li, Zhichao Jiang, Maofei Que, Yao Hu and Alexander Tuzhilin. "Dual Attentive Sequential Learning for Cross-Domain Click-Through Rate Prediction."KDD '21: Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining 2021.** [[Paper]](https://lpworld.github.io/files/kdd21.pdf)

**Dataset:**  [[Amazon dataset]](http://jmcauley.ucsd.edu/data/amazon/index_2014.html)
Due to the confidential agreement with Imhonet and Alibaba-Youku, we are not allowed to make the Imhonet and the Youku datasets publicly available. Nevertheless, you are always welcome to use our codes for the two public datasets and your own dataset.

**Please cite our KDD'21 paper if you use our codes. Thanks!** 

Author: Pan Li (https://lpworld.github.io/)

## Environment Settings
We use Tensorflow as the backend. 
- Tensorflow version:  '1.4.0'

## Example to run the codes.
The instruction of commands has been clearly stated in the codes (see the parse_args function). 

Run DASL:
```
python train.py
```

## Acknowledgement
This implementation is inspired from [DDTCDR](https://github.com/lpworld/DDTCDR).

Last Update: 2021/08/21

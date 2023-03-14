# Intelligent Adaptive Gossip-based Broadcast Protocol
***Authors**: [Zhe Ren](https://github.com/JK211), Xinghua Li, Yinbin Miao, [Zhuowen Li](https://github.com/lzwgiter), [Zihao Wang](https://github.com/wangzihao318), Ximeng Liu, Robert H. Deng* 
***Keywords**: UAVs, Gossip Protocol, Sparse Rewards, Partially Observable Markov Decision Process, Reinforcement Learning* 

## Requirements

- Python
- NS-3 (version == 3.25)

## Description

The Proof-of-concept is made up of two parts as follows:

- UAVNET environment simulated using NS-3
- Deep Reinforcement Learning code

Among them, the UAVNET environment code is `GossipEnvironment.py`, while others compose the deep reinforcement learning code.

## Installation

*Note:* The NS-3 simulation environment with python support needs to be installed, more details see: [ns-3 | a discrete-event network simulator for internet systems (nsnam.org)](https://www.nsnam.org/)

*Python Dependencies:*

- pytorch
- gym
- tqdm

## Usage

- After alter the train parameters, to train a new model:

```shell
$ python branching_dqn.py
```

- Using a pre-trained model:

```shell
$ python enjoy.py
```






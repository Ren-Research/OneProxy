# One Proxy Device Is Enough for Hardware-Aware Neural Architecture Search

### [video](https://youtu.be) | [paper](https://arxiv.org) | [website](https://github.com/Ren-Research/OneProxy) [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ren-Research/OneProxy/blob/main/example.ipynb)

[One Proxy Device Is Enough for Hardware-Aware Neural Architecture Search](https://arxiv.org/)

Bingqian Lu, Jianyi Yang, Weiwen Jiang, Yiyu Shi, [Shaolei Ren](https://intra.ece.ucr.edu/~sren/)

UC Riverside, George Mason University, University of Notre Dame

In Sigmetrics 2022

![framework](./images/sota.jpg)

Overview of NAS algorithms. Left: NAS without a supernet. Right: One-shot NAS with a supernet.


## Abstract

Convolutional neural networks (CNNs) are used in numerous real-world applications such as vision-based autonomous driving and video content analysis. To run CNN inference on various target devices, hardware- aware neural architecture search (NAS) is crucial. A key requirement of efficient hardware-aware NAS is the fast evaluation of inference latencies in order to rank different architectures. While building a latency predictor for each target device has been commonly used in state of the art, this is a very time-consuming process, lacking scalability in the presence of extremely diverse devices. In this work, we address the scalability challenge by exploiting latency monotonicity — the architecture latency rankings on different devices are often correlated. When strong latency monotonicity exists, we can re-use architectures searched for one proxy device on new target devices, without losing optimality. In the absence of strong latency monotonicity, we propose an efficient proxy adaptation technique to significantly boost the latency monotonicity. Finally, we validate our approach and conduct experiments with devices of different platforms on multiple mainstream search spaces, including MobileNet-V2, MobileNet-V3, NAS-Bench-201, ProxylessNAS and FBNet. Our results highlight that, by using just one proxy device, we can find almost the same Pareto-optimal architectures as the existing per-device NAS, while avoiding the prohibitive cost of building a latency predictor for each target device.

![flowchart](./images/flowchart.jpg)

Overview of using one proxy device for hardware-aware NAS.

## Latency Monotonicity in the Real World


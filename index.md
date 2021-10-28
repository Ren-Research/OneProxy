# One Proxy Device Is Enough for Hardware-Aware Neural Architecture Search

### [video](https://youtu.be) | [paper](https://arxiv.org) | [website](https://ren-research.github.io/OneProxy/) [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ren-Research/OneProxy/blob/main/example.ipynb)

[One Proxy Device Is Enough for Hardware-Aware Neural Architecture Search](https://arxiv.org/)

Bingqian Lu, Jianyi Yang, Weiwen Jiang, Yiyu Shi, [Shaolei Ren](https://intra.ece.ucr.edu/~sren/), Proceedings of the ACM on Measurement and Analysis of Computing Systems, vol. 5, no. 3, Dec, 2021. (**SIGMETRICS 2022**)

```BibTex
@inproceedings{
  luOneProxy2021,
  title={One Proxy Device Is Enough for Hardware-Aware Neural Architecture Search},
  author={Bingqian Lu and Jianyi Yang and Weiwen Jiang and Yiyu Shi and Shaolei Ren},
  journal = {Proceedings of the ACM on Measurement and Analysis of Computing Systems}, 
  month = Dec,
  year = 2021,
  volume = {5}, 
  number = {3},
  articleno = {34}, 
  numpages = {35},
}
```

## Hardware-aware NAS Dilemma

CNNs are used in numerous real-world applications such as vision-based autonomous driving and video content analysis. To run CNN inference on various target devices, hardware-aware neural architecture search (NAS) is crucial. A key requirement of efficient hardware-aware NAS is the fast evaluation of inference latencies in order to rank different architectures. While building a latency predictor for each target device has been commonly used in state of the art, this is a very time-consuming process, lacking scalability in the presence of extremely diverse devices.


### Overview of SOTA NAS algorithms

![framework](./images/sota.jpg)

Left: NAS without a supernet. Right: One-shot NAS with a supernet.


![nas_cost_comparison](./images/nas_cost_comparison.jpg)

Cost Comparison of Hardware-aware NAS Algorithms for 𝑛 Target Devices.


## Our approach: exploiting latency monotonicity

We address the scalability challenge by exploiting latency monotonicity — the architecture latency rankings on different devices are often correlated. When strong latency monotonicity exists, we can re-use architectures searched for one proxy device on new target devices, without losing optimality.

### Using SRCC to measure latency monotonicity

To quantify the degree of latency monotonicity, we use the metric of Spearman’s Rank Correlation Coefficient (SRCC), which lies between -1 and 1 and assesses statistical dependence between the rankings of two variables using a monotonic function. The greater the SRCC of CNN latencies on two devices, the better the latency monotonicity. SRCC of 0.9 to 1.0 is usually viewed as strongly dependent in terms of monotonicity.

We empirically show the existence of strong latency monotonicity among devices of the same platform, including mobile, FPGA, desktop GPU and CPU.

![heatmap](./images/heatmap1.jpg)

SRCC of 10k sampled models latencies in MobileNet-V2 space on different pairs of mobile and non-mobile devices.


## In the absence of strong latency monotonicity: adapting the proxy latency predictor

### SOTA latency predictors

**Operator-level latency predictor.** A straightforward approach is to first profile each operator (or each layer), and then sum all the operator-level latencies as the end-to-end latency of an architecture. Specifically, given 𝐾 operators (e.g., each with a searchable kernel size and expansion ratio), we can represent each operator using one-hot encoding: 1 means the respective operator is included in an architecture, and 0 otherwise. Thus, an architecture can be represented as x ∈ {0, 1}𝐾 ∪ {1}, where the additional {1} represents the non-searchable part, e.g., fully-connected layers in CNN, of the architecture. Accordingly, the latency predictor can be written as 𝑙 = w𝑇 x, where w ∈ R𝐾+1 is the operator-level latency vector. This approach needs a few thousands of latency measurement samples (taking up a few tens of hours).


**GCN-based latency predictor.** To better capture the graph topology of different operators, a recent study uses a graph convolutionary network (GCN) to predict the inference latency for a target device. Concretely, the latency predictor can be written as 𝑙 = 𝐺𝐶𝑁 Θ (x), where Θ is the GCN parameter learnt based on latency measurement samples and x is the graph-based encoding of an architecture.


**Kernel-level latency predictor.** Another recent latency predictor is to use a random forest to estimate the latency for each execution unit (called “kernel”) that captures different compilers and execution flows, and then sum up all the involved execution units as the latency of the entire architecture. This approach unifies different DNN frameworks, such as TensorFlow and Onnx, into a single model graph, and hence can predict latencies for models developed using different frameworks. By encoding an architecture based on the execution units, we can also transform the latency predictor into a linear one: 𝑙 = w𝑇 x where w is the vector of latencies for different execution units and x denotes the number of each execution unit included in an architecture. Thus, an “execution unit” in nn-Meter is conceptually equivalent to a searchable operator in the operator-level latency predictor.


### AdaProxy for boosting latency monotonicity

Even though two devices have weak latency monotonicity, it does not mean that their latencies for each searchable operator are uncorrelated; instead, for most operators, their latencies can still be roughly proportional. The reason is that a more complex operator with higher FLOPs that is slower (say, 2x slower than a reference operator) on one device is generally also slower on another device, although there may be some differences in the slow-down factor (say, 2x vs. 1.9x). This is also the reason why some NAS algorithms use the device-agnostic metric of architecture FLOPs as a rough approximation of the actual inference latency. If we view proxy adaptation as a new learning task, this task is highly correlated with the task of building the proxy device’s latency predictor, and such correlation can greatly facilitate transfer learning. We exploit the correlation among devices and propose efficient transfer learning to boost the otherwise possibly weak latency monotonicity for a target device.

<p align="center">
  <img src="./images/heatmap_s5e_cross.jpg">
</p>

In the MobileNet-V2 space, with S5e as default proxy device


![nasbench_heatmap](./images/nasbench_heatmap.jpg)

In the NAS-Bench-201 search space on CIFAR-10 (left), CIFAR-100 (middle) and ImageNet16-120 (right) datasets, with Pixel3 as our proxy device


![nasbench_heatmap](./images/nasbench_heatmap.jpg)

In the FBNet search spaces on CIFAR-100 (left) and ImageNet16-120 (right) datasets, with Pixel3 as our proxy device


<p align="center">
  <img src="./images/heatmap_rice_eagle.jpg">
</p>

SRCC for various devices in the NAS-Bench-201 search space with latencies collected from [19, 29, 49, 50]


## Using one proxy device for hardware-aware NAS

![flowchart](./images/flowchart.jpg)


### One proxy for hardware-aware NAS

![ea_models](./images/ea_models.jpg)

![exhaustive_models](./images/exhaustive_models.jpg)

![rice_nasbench_cifar10](./images/rice_nasbench_cifar10.jpg)

Exhaustive search results for different target devices on NAS-Bench-201 architectures (CIFAR-10 dataset). Pixel3 is the proxy.


## Public latency datasets used in this work

[HW-NAS-Bench: Hardware-Aware Neural Architecture Search Benchmark](https://github.com/RICE-EIC/HW-NAS-Bench)

[Eagle: Efficient and Agile Performance Estimator and Dataset](https://github.com/zheng-ningxin/brp-nas)

[nn-Meter: towards accurate latency prediction of deep-learning model inference on diverse edge devices](https://github.com/microsoft/nn-Meter)

[Once for All: Train One Network and Specialize it for Efficient Deployment](https://github.com/mit-han-lab/once-for-all)

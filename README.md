# network_width_and_transfer_learning

Experiments on the relationship between hidden layer width and transfer learning inspired by https://arxiv.org/pdf/1909.11572.pdf

Do wider networks really learn better features for transfer learning, or is it just that narrow networks contain information bottlenecks?

By comparing the effect of network size on the fine tuning of a network for a novel task using transfer learning class to the effect of network size on a ranomly initialized network which is then fine tuned on the novel task, we show that information bottle necks could explain why wider networks outperform shallow ones on fine tuning tasks, even if better features are not learned.

| Notebook     |      Description      |     |
|:----------|:-------------|------:|
| [Run Sweep](https://github.com/iantheconway/network_width_and_transfer_learning/sweep.ipynb)  | Use Weights and Biases to run a sweep of different network widths|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iantheconway/network_width_and_transfer_learning/blob/master/sweep.ipynb) |



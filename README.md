### Cartpole魔改-应用于卷线器

需要参数：`未知`

需要结果：`动作`

使用MLP结合Q-learning，写出了类似于DQN的强化学习程序。

MLP的作用：给出期望Q值。当我有向左移动和向右移动两个动作时，对每个动作预测一个期望Q值。

Q-learning：给出targetQ值，成为targets，和inputs打包塞到MLP中；maxQValue()选择出Q值更大的决策；epsilonGreedyPolicy()根据状态state通过MLP前向传播得到outputs[0]和outputs[1]，并返回Q值更大的动作作为决策。






























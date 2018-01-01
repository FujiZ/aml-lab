针对Q-learning、DQN以及Improved DQN，
我分别分别实现了对应的Helper class，
因此训练和使用均可采用统一的接口。

举例来说，对于Q-learning，若我们想运行CartPole，
先训练5000轮，然后再跑一轮测试，可以执行以下代码：

```Python
import myQLearning

pole = myQLearning.CartPole()
pole.train(5000)
pole.play()
```

Helper class被命名为CartPole、MountainCar以及Acrobot，
分别对应gym中的三个实验环境。

需要注意的是，对于helper的train是可重入的，
即我们可以先训练5000轮，然后测试一下看看效果，
在此基础上再训练1000轮。

另外，根据[Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)：

> Modules should have short, all-lowercase names. 
> Underscores can be used in the module name if it improves readability.
> Python packages should also have short, all-lowercase names, although the use of underscores is discouraged.

因此在实验说明中类似MyDQN.py这样的命名实际上是不符合规范的，
希望在以后布置作业的时候能稍微注意一下代码规范的问题。


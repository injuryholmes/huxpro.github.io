---
layout: post
title: "cs231n Lecture 3<br>Optimization Notes<br>优化笔记"
date: 2018-01-28
author: "injuryholmes"
catalog: false
header-img: "img/post-bg-cs231n.jpg"
tags:
    - optimization
    - gradient descent
    - numerical gradient
    - analytical gradient
---

cs231N是斯坦福大学的卷积神经网络课程，我通过博客记录自己的学习心得，对于原文中一些难点做了一下详细的解释。希望能给看文章的你带来帮助。

*为了方便大家对照原文，我尽量在直白翻译和自己的理解之间做权衡，所以文章读起来会有些琐碎。*

*本博客仅做参考，如有措辞不当或理解错误，欢迎大家留言指正。*

更多原文内容请查看[官网](http://cs231n.github.io/optimization-1/)。

- [简介 Introduction](#intro)
- [损失函数可视化 Visualizing the loss function](#vis)
- [优化 Optimization](#optimization)
  - [Strategy #1: Random Search](#opt1)
  - [Strategy #2: Random Local Search](#opt2)
  - [Strategy #3: Following the gradient](#opt3)
- [梯度计算 Computing the gradient](#gradcompute)
  - [Numerically with finite differences](#numerical)
  - [Analytically with calculus](#analytic)
- [Gradient descent](#gd)
- [总结 Summary](#summary)

<a name='intro'></a>

### 简介 Introduction

在前面的章节中，我们介绍了图片分类任务中的两个关键部分：

1. 用含有参数的得分函数（score function，例如，线性函数）预测不同类别的分数。
2. 损失函数根据训练数据中预测分数与真实标签的一致程度来衡量一组特定参数的好坏。常见的有Softmax / SVM。

具体来说，回忆一下线性函数的形式为：$f(x_i, W) =  W x_i$，损失函数SVM为：


$$
L = \frac{1}{N} \sum_i \sum_{j\neq y_i} \left[ \max(0, f(x_i; W)_j - f(x_i; W)_{y_i} + 1) \right] + \alpha R(W)
$$


我们已经知道，当我们设置 W 使其预测的标签和真实的标签一致的比例越高，模型的损失就越低。 本文要介绍第三个也是最后一个关键部分：优化。 优化是找到使损失函数最小化的一组参数 W 的过程。

​	一旦我们理解了这三个关键部分如何相互作用，我们将重新回顾第一部分（参数化函数映射预测分数），并将其扩展到比线性映射复杂得多的函数：整个神经网络，然后是卷积神经网络。 损失函数和优化过程将保持相对不变。

<a name='vis'></a>

### 损失函数可视化 Visualizing the loss function

​	我们平时看到的损失函数通常是在高维的空间上定义的（例如，在CIFAR-10中，线性分类器的权重矩阵大小为 $[10×3073]$ ，总共有30,730个参数），这使得它们很难被可视化。但是，我们仍然可以通过沿着直线（1维）或沿着平面（2维）切割高维空间的方法来获得一些直观的理解。例如，我们可以生成一个随机的权重矩阵 W（它对应于二维空间中的一个点。这和我们在lecture 2中说道的一样，所有的高维空间里的物体，我们都可以看成二维空间中的一个点）。然后，我们沿着一条射线前进并记录损失函数值。也就是说，我们可以产生一个随机方向 $W_1$ ，并通过对不同的 $a$ 值计算 $L* (W + aW_1)$ ，来计算沿这个方向的损失。这个过程产生一个简单的绘图，其中以 $a$ 为 x 轴，而损失函数 $L$ 为 y 轴。我们也可以再二维空间执行相同的过程。此时我们的损失为 $L * (W + aW_1 + bW_2)$ 。在图中，a，b对应于x轴和y轴，并且损失函数的值可以用颜色的深浅可视化：

![visLoss](/img/in-post/2018-01-28-optimization/visLoss.png)

举个例子大家就懂了：

​	上面的三张图是使用 Multiclass SVM（没有正规化损失部分）。左图和中图是针对一张图片计算的损失值，分别用一维和二维的形式表示。右图是针对100张CIFAR-10中的图片计算的损失的平均值，通过二维的形式表示。可见左图，损失值最低的地方在第二个格子末端。在二维图中，我们定义蓝色表示损失低，红色表示损失高。注意，左图被分割成不同损失值区段的这种结构，下文会解释原因。而右图呈现美丽的圆形是因为它取的是分类器针对多个图片损失值的平均值，大家可以想象右图是多张中图以不同旋转角度重叠在一起之后我们看到的样子。

​	对于左图这种被分割成不同损失值区段的结构，我们从数学角度分析一波：


$$
L_i = \sum_{j\neq y_i} \left[ \max(0, w_j^Tx_i - w_{y_i}^Tx_i + 1) \right]
$$


​	从等式可以清楚地看出，每个例子的数据损失必大于零（0是阈值）。举个例子，一个包含三个一维点和三个类的简单数据集。 完整的SVM损失（没有正则化）如下：


$$
\begin{align}
L_0 = & \max(0, w_1^Tx_0 - w_0^Tx_0 + 1) + \max(0, w_2^Tx_0 - w_0^Tx_0 + 1) \\\\
L_1 = & \max(0, w_0^Tx_1 - w_1^Tx_1 + 1) + \max(0, w_2^Tx_1 - w_1^Tx_1 + 1) \\\\
L_2 = & \max(0, w_0^Tx_2 - w_2^Tx_2 + 1) + \max(0, w_1^Tx_2 - w_2^Tx_2 + 1) \\\\
L = & (L_0 + L_1 + L_2)/3
\end{align}
$$


​	咱们简单的可视化一下：

![visLossSum](/img/in-post/2018-01-28-optimization/visLossSum.png)

​	首先左图中有红、蓝、绿三条铰链，大家还记得铰链损失函数吗？不记得的话，看我[上一篇文章](http://injuryholmes.me/2018/01/21/linear-classification/)复习一下吧。左图的 x 轴是权重矩阵 W，y轴是损失值。每个例子的总损失函数是本例子对应多个错误类别标签的铰链损失的总和。完整的SVM数据丢失是这种形状的30,730维版本。

​	顺便说一下，你可能已经从上面SVM的碗形外观中猜测出SVM是一个凸函数。有大量的文献致力于有效地最小化这种类型的函数，斯坦福类也有关于凸优化的[课程](http://stanford.edu/~boyd/cvxbook/)。一旦我们将得分函数扩展到神经网络，我们的目标函数将变成非凸的，而上面的可视化也将不会具有碗形，而是复杂、凹凸不平的形状。

​	大家注意，由于 max 的性质，导致损失函数在转折点不可微分。 因为在这些转折处没有定义梯度。 然而，次梯度（subgradient）仍然存在，而且我们常常会用到。

<a name='optimization'></a>

### Optimization

​	重申一下，损失函数使我们可以量化任何特定的权重集 W 的好坏。优化的目标是找到使损失函数最小的 W 。我们将循序渐进地学习一种优化损失函数的方法。对于那些有经验的人来说，这部分可能看起来很奇怪，因为我们将使用的示例（SVM损失）是一个凸优化问题。但请记住，我们的目标是最终优化神经网络，所以不会使用在凸优化中的特定工具。

<a name='opt1'></a>

#### Strategy #1: A first very bad idea solution: Random search

​	既然检查给定的参数集合 W 非常简单，那么我们想到的第一个（非常糟糕的）想法就是简单地尝试许多不同的随机权重，并跟踪哪些是最好的。 这个过程可能如下所示：

```python
# assume X_train is the data where each column is an example 维度(3073, 50,000)
# assume Y_train are the labels 维度(50000, )
# assume the function L evaluates the loss function

bestloss = float("inf") # Python assigns the highest possible float value
for num in xrange(1000):
    W = np.random.randn(10, 3073) * 0.0001 # generate random parameters
    loss = L(X_train, Y_train, W) # get the loss over the entire training set
    if loss < bestloss: # keep track of the best solution
        bestloss = loss
        bestW = W
    print 'in attempt %d the loss was %f, best %f' % (num, loss, bestloss)

# prints:
# in attempt 0 the loss was 9.401632, best 9.401632
# in attempt 1 the loss was 8.959668, best 8.959668
# in attempt 2 the loss was 9.044034, best 8.959668
# in attempt 3 the loss was 9.278948, best 8.959668
# in attempt 4 the loss was 8.857370, best 8.857370
# in attempt 5 the loss was 8.943151, best 8.857370
# in attempt 6 the loss was 8.605604, best 8.605604
# ... (trunctated: continues for 1000 lines)
```

在上面的代码中，我们尝试了几个随机权重矩阵 W，其中一些比其他的更好。 我们可以通过这个搜索找到所有随机矩阵中最好的 W，并在测试集上进行测试：

```python
# Assume X_test is [3073 x 10000], Y_test [10000 x 1]
scores = Wbest.dot(Xte_cols) # 10 x 10000, the class scores for all test examples
# find the index with max score in each column (the predicted class)
Yte_predict = np.argmax(scores, axis = 0)
# and calculate accuracy (fraction of predictions that are correct)
np.mean(Yte_predict == Yte)
# returns 0.1555
```

​	用最好的 W 可以得到约15.5％的精度。鉴于完全随机的猜测类只能达到10％，对于随机搜索这种无脑的解决方案来说这不是一个非常糟糕的结果。
​	核心思想：迭代优化（iterative refinement）。 当然，我们可以做得更好。但这个简单的例子告诉我们，核心思想是找到最好的权重集 W 是一个非常困难甚至是不可能的问题（特别是一旦 W 包含整个复杂神经网络的所有权重）。但是将一组特定的权重 W 稍微调整的好一点就相对简单一点了。 所以，我们的方法是从一个随机的 W 开始，然后迭代地优化它，每次都稍微好一点。

 	举个例子：你是一个被蒙住眼睛的徒步旅行者。 在山路上徒步，并试图达到底部。 在CIFAR-10的例子中，由于 W 的尺寸是10×3073，所以山是30,730维。在山上的每一个点我们都拥有特定的损失（海拔高度）。

<a name='opt2'></a>

#### Strategy #2: Random Local Search

​	你可能会想到的第一个策略是试图在一个随机的方向上小心翼翼的试探，然后只有感觉到这是下坡的时候才采取这一步。

<img src="/img/in-post/2018-01-28-optimization/tryStep.jpg" width="30%">

具体来说，我们将从随机 W 开始，产生随机扰动 $\delta W$ 。如果 $W+\delta W$ 对应的损失较低的话，我们就执行更新。 此过程的代码如下所示：

```python
W = np.random.randn(10, 3073) * 0.001 # generate random starting W
bestloss = float("inf")
fir i in xrange(1000):
    step_size = 0.001
    Wtry = W + np.random.randn(10, 3073) * step_size
    loss = L(Xtr_cols, Ytr, Wtry)
    if loss < bestloss:
        W = Wtry
        bestloss = loss
    print 'iter %d loss is %f' % (i, bestloss)
```

​	我们同样跑1000次。该方法在测试集上实现了21.4％的准确性。 比随机强，但是仍然很浪费，计算量很大。

<a name='opt3'></a>

#### Strategy #3: Following the Gradient

​	上一个例子中，我们试图在权重空间中找到一个方向来改善我们的权重（更低的损失值）。而事实上，我们不需要随机搜索一个方向，再判断它的好坏。我们可以直接计算出最好的方向，这个方向在数学上保证是最陡峭的下降方向。这个方向将与损失函数的梯度（**gradient**）有关。好比我们爬山的时候，感觉最陡峭的方向。

​	在一维函数中，斜率是函数在任何点的瞬时变化率。梯度是函数的斜率的泛化，比如我们的 W 是高维的，所以对应的，从单个斜率变成一组斜率组成的向量。换句话说，梯度就是空间中每一个维度的斜率（slope或者derivative）组成的向量。高中学过的倒数就是一个一维函数梯度的例子。


$$
\frac{df(x)}{dx} = \lim_{h\ \to 0} \frac{f(x + h) - f(x)}{h}
$$


​	当函数采用一个数字向量而不是单一个数字的时候，我们称每一维度的倒数为偏导数（prtial derivatives），而梯度就是每个维度上的偏导数共同组成的一个向量。

<a name='gradcompute'></a>

### Computing the gradient

​	有两种计算梯度的方法，一个缓慢的，近似的但容易的方法——数值梯度（numerical gradient），以及一个快速，准确，但更容易出错的方法，需要微积分——分析梯度（analytic gradient）。 

<a name='numerical'></a>

#### Computing the gradient numerically with finite differences

​	上面给出的公式允许我们以数值方式计算梯度。 下面是一个函数，它计算函数`f`在`x`处的导数。

```python
def eval_numerical_gradient(f, x):
    """
    a native implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """
    
    fx = f(x) # evaluate function value at original point
    grad = np.zeros(x.shape)
    h = 0.00001
    
    # iterate over all indexes in x
    it = np.nditer(x, flags=['muilti_index'], op_flags=['readwrite'])
    while not it.finished:
        # evaluate function at x+h
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h # increment by h
        fxh = f(x) # evaluate f(x + h)
        x[ix] = old_value # restore to previous value (very important!)
       	
        # compute the partial derivate
        grad[ix] = (fxh - fx) / h # the slope
        it.iternext() # step to next dimention
    return grad
```

按照上面给出的梯度公式，上面的代码逐个遍历所有维度，沿该维度做一个小的变化`h`，并通过查看函数的变化来计算损失函数沿着该维度的偏导数。 变量`grad`即是完整的梯度。

​	**实际情况**。 请注意，在数学公式中，我们希望`h`趋向于零来计算梯度，但实际上通常使用非常小的值（例如在示例中看到的1e-5）就足够了。 当然，最理想的情况是，在计算机算力允许的情况下，挑选尽量小的步长。 另外，在实践中，使用中心差分公式计算数值梯度通常更好：$[f(x + h)-f(x-h)] / 2h$。 详情请参阅[维基](https://en.wikipedia.org/wiki/Numerical_differentiation)。

​	我们可以使用上面的梯度计算函数来计算给定函数任意点处的梯度。 我们来看一个例子，给定一个权重矩阵`W`，计算CIFAR-10损失函数的梯度：

```python
# to use the generic code above we want a function that takes a single argument
# (the weight in our case) so we close over X_train and Y_train

def CIFAR10_loss_fun(W):
    return L(X_train, Y_train, W)

W = np.random.rand(10, 3073) * 0.001 # random weight vector
df = eval_numerical_gradient(CIFAR_loss_fun, W) # get the gradient
```

梯度告诉我们沿着每个维度的损失函数的斜率，**它给我们描述了山谷的地形图**，我们可以用它来进行更新：

```python
loss_original = CIFAR10_loss_fun(W) # the original loss
print 'original loss: %f' % (loss_original, )

# lets see the effect of multiple step sizes
for step_size_log in [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]:
    step_size = 10 ** step_size_log
    W_new = W - step_size * df # new position in the weight space
    loss_new = CIFAR10_loss_fun(W_new)
    print 'for step size %f new loss: %f' % (step_size, loss_new)
    
# prints:
# original loss: 2.200718
# for step size 1.000000e-10 new loss: 2.200652
# for step size 1.000000e-09 new loss: 2.200057
# for step size 1.000000e-08 new loss: 2.194116
# for step size 1.000000e-07 new loss: 2.135493
# for step size 1.000000e-06 new loss: 1.647802
# for step size 1.000000e-05 new loss: 2.844355
# for step size 1.000000e-04 new loss: 25.558142
# for step size 1.000000e-03 new loss: 254.086573
# for step size 1.000000e-02 new loss: 2539.370888
# for step size 1.000000e-01 new loss: 25392.214036
```

**更新梯度方向为负**。在上面的代码中，请注意，为了计算`W_new`，我们对梯度`df`的负方向进行更新，因为我们希望我们的损失函数减少而不是增加。

**步长的影响**。梯度告诉我们函数的增长速度最快的方向（损失增加、或者减少最快的方向），但是并没有告诉我们该沿着这个方向走多远。 正如我们后面将会看到的那样，选择步长（也称 learning rate）将成为训练神经网络中最重要的（也是最令人头痛的）超参数设置之一。 在我们蒙着眼睛的寻找山丘最低点的比喻中，我们感觉到我们的脚下方的山坡向某个方向倾斜，但是我们应该采取的步长是不确定的。 如果我们谨慎地调整脚步，我们可以持续地取得非常小的进展（这相当于有一个小步幅）。 相反，我们可以选择做出一个大的，自信的步骤，试图更快地下降，但这可能没有回报。 正如上面的代码示例，在某个时刻，采取更大的步骤会导致更高的损失，因为我们`overstep`了。

​	让我们可视化步长的影响，如下图所示。

<img src="/img/in-post/2018-01-28-optimization/visStepsize.png" width="40%">

​	我们从某个特定的 W 点开始（别忘了每一个点代表某一个权重矩阵W），计算该 W 的梯度（或者说梯度的反向（减小的方向）—— 白色箭头），它告诉我们损失函数最急剧下降的方向。 小步骤可能会导致持续但缓慢的进展，我们会慢慢地往深蓝色的区域移动。 大步骤可以带来更好的进展，但风险更大。 请注意，最终，如果步长过大，我们会`overshoot`，并使损失更严重，例如，我们可能直接从浅蓝色地方跑到绿色的区域。 步长将成为我们必须仔细调整的最重要的超参数之一。

​	**效率问题**。你可能已经注意到，这种数值法计算梯度的时间复杂度是和参数（维度）的多少成线性比例。 在我们的例子中，我们总共有30730个参数，因此必须对损失函数进行30,731（origin + 30730 new）次评估，以评估梯度并仅执行单个参数更新。 这个问题只会变得更糟，因为现代的神经网络常常地拥有数千万个参数。 显然，这个策略是不可扩展的，我们需要更好的方法。

​	<a name='analytic'></a>

#### Computing the gradient analytically with Calculus

​	我们把上述计算梯度的方法叫做numerical gradient，理解起来很简单，但缺点是它是近似的（因为我们必须选择小的h值，而真正的梯度定义为当h趋向于0的极限）， 而且算力昂贵。 计算梯度的第二种方法是使用微积分（calculus）进行分析计算，这使得我们能够得出梯度的直接公式（无近似值），而且速度也非常快。 但是，与数值法计算梯度不同，`analytic gradient` 法更容易实现的代价就是更容易出错，所以在实践中，我们使用 `analytic gradient` 并将其与 `numerical gradient` 进行比较以检查实现的正确性。 这被称为梯度检查（gradient check）。

​	让我们举个svm的例子：


$$
L_i = \sum_{j\neq y_i} \left[ \max(0, w_j^Tx_i - w_{y_i}^Tx_i + \Delta) \right]
$$


我们求该函数对 $W_{y_i}$ 的倒数。我们可以得到如下的式子：


$$
\nabla_{w_{y_i}} L_i = - \left( \sum_{j\neq y_i} \mathbb{1}(w_j^Tx_i - w_{y_i}^Tx_i + \Delta > 0) \right)x_i
$$


解释一下：**

**把 $W_{y_i}$ 看做自变量，其他的都看成常量。损失函数可以看成一元一次方程，自变量的系数是 $-x_i$ （常量）**

其中 $\mathbb{1}$ 是指标函数，如果内部条件为真，则为1，否则为0。这是损失函数的max的性质导致。也就是说，我们只需要计算不符合`margin`的类（也就是那些导致损失函数增加的类）即可。同时我们也可以发现一个直观的语义学上的理解：**损失函数在正确类上的倒数（梯度）和损失的类以及该类的损失值大小有关，最后的梯度是该类和该类损失值相乘的负数。**

用山谷的例子来理解：**我们往山谷底部走的最快的路线，是往山顶走的最快的路的反方向。**这话一听上去有点脱裤子放屁，但是大家好好理解，其实这就是这个公式本身诉说的。

​	上面讲的都只是损失函数相对于正确类的梯度（也就是说我们只求了权重矩阵中$W_{y_i}$ 行的梯度）。 对于 $j≠yi$ 的其他行，梯度为：


$$
\nabla_{w_j} L_i = \mathbb{1}(w_j^Tx_i - w_{y_i}^Tx_i + \Delta > 0) x_i
$$


**同理：**

把 $W_j$ 看做自变量，其他的都看成常量。损失函数可以看成一元一次方程，自变量的系数是 $x_i$ （常量）

​	大家注意一下，之前正确类的系数是负数，而错误类的系数是正数，这是因为原函数的意义是损失函数，当然在正确类上损失应该是呈减小趋势，（直到山谷底端）；而在错误类上，损失是呈增加趋势的。我们之后要做的，就是求出整个权重矩阵的梯度之后，往负的方向前进。

有了公式，用代码实现一下就没问题了。

<a name='gd'></a>

### Gradient Descent 梯度下降

​	现在我们可以计算损失函数的梯度，重复评估梯度然后执行参数更新的过程称为梯度下降（Gradient Descent）。 它的原始版本如下：

```python
# Vanilla Gradient Descent

while True:
    weights_grad = evaluate_gradient(loss_fun, data, weights)
    weights += -step_size * weights_grad # perform parameter update
```

这个简单的 `while` 循环是所有神经网络库的核心。 还有其他的方法来执行优化（例如，LBFGS），但是梯度下降是目前为止优化神经网络损失函数最常见和最成熟的方式。 在整个课程教学中，我们会对这个循环的细节（例如更新方程的确切细节）进行一些细致的分析，但是核心思想就是梯度下降直到我们对结果满意为止。	

**Mini-batch gradient descent**。 在大规模应用（如ILSVRC挑战）中，训练数据可能具有数百万个。 因此，很容易产生低效的更新，比如，当我们计算了整个训练集的损失函数梯度，却只更新其中一个参数。 解决这个问题的一个非常普遍的方法是计算批量（batch）训练数据的梯度。 例如，在现有技术的ConvNets中，典型的批量参数包含来自整个120万训练集的256个例子。 然后使用该批次执行参数更新：

```python
# Vanilla Minibatch Gradient Descent

while True:
    data_batch = sample_training_data(data, 256) # sample 256 examples
    weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
    weights += -step_size * weights_grad
```

​	我们可以这么做是因为训练数据中的例子是相关的。 我们举一个极端的例子，ILSVRC中的所有120万个图像实际上仅含有1000个独特图像（每个类别一个，重复1200次）的副本组成。 那么很明显，我们计算的所有1200个相同副本的梯度都是相同的。那我们计算所有120万个图像上的数据损失时，和我们只计算一小部分1000个的损失是一样的。在实践，数据集不会包含重复的图像，但一个小批量的梯度是一个完整的目标的梯度很好的近似。 因此，通过评估小批量梯度来执行更频繁的参数更新，在实践中可以实现更快的收敛。

​	这种情况的极端例子是小批量仅包含一个示例。这个过程被称为随机梯度下降**Stochastic Gradient Descent (SGD)**（或 on-line 梯度下降）。这是相对较少见的，因为在实践中，由于矢量化的代码优化，对于100个示例计算梯度，比对于一个示例计算100次的梯度更快。即使SGD理论上一次只用一个例子来评估梯度，但人们现在在说SGD的时候，往往默认了minibatch的使用。（即谈及 Minibatch Gradient Descent （MGD），或提到 batch gradient descent（BGD）时，也会听到人们使用术语SGD指代，但此时通常假定使用小批量）。mini-batch的大小是一个超参数，但交叉验证并不常见。它通常基于内存约束（如果有的话）或设置为某个值，例如32，64或128。我们在实践中使用2的幂，因为许多向量化的操作实现在输入大小为2的幂的情况下更快。

<a name='summary'></a>

### Summary

<img src="/img/in-post/2018-01-28-optimization/dataflow.jpeg">

数据集 `(x，y)` 是给定的。权重 `W` 从随机开始，可以改变。得分函数 `f` 计算类别的预测分数。损失函数包含两个部分：第一部分表示的是预测计算得分 `f` 和真实标签 `y` 之间的差异性。第二部分正则化损失是权重的函数。在梯度下降期间，我们计算权重的梯度，并在梯度下降期间使用它们执行参数更新。

本文小结：

- 我们将损失函数比作一个高维的爬山模型，我们蒙着眼睛，试图达到最低点。特别是，我们看到SVM函数是分段线性，整体是碗形的。
- 我们通过迭代来优化损失函数，我们从一个随机的权重 `W` 开始，一步一步地优化，直到损失最小化。
- 我们看到梯度最陡的上升方向，我们讨论了一种简单但低效的方法，用有限差分近似（即取非常小的 `h` 值进行导数操作）对其进行数值计算。
- 我们看到，参数更新需要设置步长（学习率）。如果它太低，进展是稳定的，但速度慢。如果它太高，进展会更快，但风险更大。我们将在以后的章节中更详细地探讨这种权衡。
- 我们讨论了 `numerical gradient` 和 `analytic gradient` 之间的权衡问题。`numerical gradient` 很简单，但是计算是近似的、昂贵的。`analytic gradient` 是精确的、快速的，但更容易出错，因为它需要用数学推导梯度。因此，在实践中，我们总是使用 `analytic gradient`，然后用 `numerical gradient` 执行梯度检查。
- 我们引入了 `gradient descent` 算法，迭代计算梯度并在循环执行参数更新。

预告：

本节的核心内容是计算损失函数的权重的梯度（并对其有一定的直观理解）。之后设计、训练和理解神经网络需要对梯度有感性的认识。在下一节中，我们将分析 `chain rule` （链规则），或称为 `backpropagation` （反向传播）。这将使我们能够更有效地优化各种神经网络的各式各样的损失函数，当然也包括我们的卷积神经网络。

版权声明：

- 自由转载-非商用-非衍生-保持署名（[创意共享4.0许可证](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.zh)）




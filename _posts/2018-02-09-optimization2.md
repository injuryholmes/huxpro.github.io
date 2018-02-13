---
layout: post
title: "cs231n Lecture 4<br>Backpropagation Notes<br>反向传播笔记"
date: 2018-02-09
author: "injuryholmes"
catalog: false
header-img: "img/post-bg-cs231n.jpg"
tags:
    - forward pass
    - back propagation
    - chain rule
    - gradient 
    - staged computation
---

cs231N是斯坦福大学的卷积神经网络课程，我通过博客记录自己的学习心得，对于原文中一些难点做了一下详细的解释。希望能给看文章的你带来帮助。

*为了方便大家对照原文，我尽量在直白翻译和自己的理解之间做权衡，所以文章读起来会有些琐碎。*

*本博客仅做参考，如有措辞不当或理解错误，欢迎大家留言指正。*

更多原文内容请查看[官网](http://cs231n.github.io/optimization-2/)。

- [Introduction](#intro)
- [Simple expressions, interpreting the gradient](#grad)
- [Compound expressions, chain rule, backpropagation](#backprop)
- [Intuitive understanding of backpropagation](#intuitive)
- [Modularity: Sigmoid example](#sigmoid)
- [Backprop in practice: Staged computation](#staged)
- [Patterns in backward flow](#patters)
- [Gradients for vectorized operations](#mat)
- [Summary](#summary)

<a name='intro'></a>

### Introduction

在本节中，我们将学习反向传播，这是通过递归使用`chian rule`来计算梯度的一种方式。了解这个过程对于理解和有效开发设计和调试神经网络至关重要。

**problem statement.** 本文讨论的核心问题就是给定函数 $f(x)$ ，输入 $x$ 是一个向量。我们要计算的就是 $f(x)$ 关于 $x$ 的导数 $\nabla f(x)$ 。

**Motivation** 

回想一下，我们对这个问题感兴趣的主要原因是，在神经网络中，$ f $ 就是损失函数 $L$ ，输入包括训练集 $x$ 和权重矩阵 $W$ 。我们举SWM的例子，那么输入就是训练数据 $(xi, yi), i = 1 … N$ 以及权重 $W$ 和偏差 $b$ 。在机器学习中，我们通常将训练数据视为给定的，将权重看作是我们控制的变量。因此，尽管我们可以很容易地使用`back prop`来计算输入样本 $xi$ 上的梯度，但实际上我们通常只计算参数 $W, b$ 的梯度，并对它们进行参数更新。不过 $xi$ 上的梯度也可能是有用的，我们之后会讲到，比如可以用它们可视化和解释神经网络在做什么。

<a name='grad'></a>

### Simple expressions and interpretation of the gradient

让我们从简单的例子开始。考虑两个数的乘法函数 $f(x,y) = xy$ 。推导任一输入的偏导数是一个简单的微积分问题：


$$
f(x,y) = x y \hspace{0.5in} \rightarrow \hspace{0.5in} \frac{\partial f}{\partial x} = y \hspace{0.5in} \frac{\partial f}{\partial y} = x
$$


​	**Interpretation** 请记住导数定义公式的意义：它们表示函数相对于围绕特定点附近无限小区域的变量的变化率：
$$
\frac{df(x)}{dx} = \lim_{h\ \to 0} \frac{f(x + h) - f(x)}{h}
$$


​	注意等式左侧的除法与右侧的除法不同。左边的这个符号表示对函数 $f$ 执行运算符 $\frac{d}{dx}$ ，求得导数，右边就是单纯的除法。当 $h$ 很小时，函数近似一条直线，导数是它每个点的斜率。换句话说，每个变量的导数告诉你整个表达式对其值的敏感度。例如，$x = 4, y = -3$ ，那么 $f(x,y)= - 12$ ，并且 $\frac{\partial f}{\partial x} = -3$ 。这表示如果我们要把这个点的 $x$ 值增加一点，那么对整个表达式的影响就是减少它的三倍。这可以通过重新排列上面的等式 $f(x + h) = f(x) + h \frac{df(x)}{dx}$ 来看出。同理，$\frac{\partial f}{\partial y} = 4$ ，如果将 $y$ 值增加一点 $h$ 也会增加函数的输出，并增加 $4h$ 。

> The derivative on each variable tells you the sensitivity of the whole expression on its value.
>
> 函数倒数表示的是函数对于某点变化的敏感程度。
>
> 敏感程度反应在该点变化对函数值变化的剧烈程度。

如前所述，梯度 $\nabla f $ 是所有偏导数组成的向量，所以我们有 $\nabla f = [\frac {\partial f} {\partial x}，\frac {\partial f} {\partial y}] = [y，x]$ 。尽管梯度本质上是一个向量，但为简单起见，我们通常会说 **the gradient on x** 而不是 **the partial derivative on x** 。

我们也可以求`add`的导数：


$$
f(x,y) = x + y \hspace{0.5in} \rightarrow \hspace{0.5in} \frac{\partial f}{\partial x} = 1 \hspace{0.5in} \frac{\partial f}{\partial y} = 1
$$


以及`max`的导数：


$$
f(x,y) = \max(x, y) \hspace{0.5in} \rightarrow \hspace{0.5in} \frac{\partial f}{\partial x} = \mathbb{1}(x >= y) \hspace{0.5in} \frac{\partial f}{\partial y} = \mathbb{1}(y >= x)
$$


由于`max`只对较大的数敏感，所以导数在一个输入上是1，而在另一个输入上是0。举个例子，如果输入是 $x = 4, y = 2$，那么最大值是4，即`max` 对$y$ 的值不敏感，如果我们增加一个微小的 $h$ ，函数会继续输出4，所以 $y$ 的梯度为零：没有效果。当然，如果我们增加 $y$ 大于2，那么 $f$ 的值将会改变，但是导数的定义是 $\lim{h}→0$ 对函数的影响，它们只是对输入信号的极小变化提供了信息。

<a name='backprop'></a>

### Compound expressions with chain rule

​	现在看一个更复杂的表达式，如 $f(x,y,z)=(x + y)z$ 。虽然这个表达式能直接求导，但是我们会从`backprop`的角度分析。这个表达式可以分解为两个表达式：$q = x + y$ 和 $f = q z$ 。此外，我们知道如何分别计算两个表达式的导数。$f$ 是 $qz$ 的乘积，所以 $\frac {\partial f} {\partial q} = z$ ，$\frac{\partial f}{\partial z} = q$ 。而 $q = x + y$ ，所以 $\frac{\partial q}{\partial x} = 1, \frac{\partial q}{\partial y} = 1 $ 。实际上，我们不关心中间值 $q$ 的梯度，也就是说$\frac {\partial f} {\partial q} = z$ 这个值其实没有什么用，相反，我们最终感兴趣的是 $f$ 相对于其输入 $x,y,z$ 的梯度。我们使用`chain rule` 计算这些最终需要的梯度。例如，$\frac {\partial f} {\partial x} = \frac {\partial f} {\partial q} \frac {\partial q} {\partial x}$ 。让我们看一个例子：

```python
# set some inputs
x = -2; y = 5; z = -4

# perform forward pass
q = x + y 	# q becomes 3
f = q * z 	# f becomes -12

# perform the backward pass (backpropagation) in reverse order:
# first backprop through f = q * z
dfdz = q # df/dz = q, so gradient on z becomes 3
dfdq = z # df/dq = z, so gradient on q becomes -4
# now backprop through q = x + y
dfdx = 1.0 * dfdq # dq/dx = 1. And the multiplication here is the chain rule!
dfdy = 1.0 * dfdq # dq/dy = 1
```

我们把偏导数存在了$[dfdx，dfdy，dfdz]$ 中，它们描述了函数对于变量 $x，y，z$ 变化的敏感性。这是反向传播最简单的例子。强调一点，之后为了简便，我们就不写 $dfdq$ 了，只写 $dq$ ，并且总是假设梯度是相对于最终输出的。

我们可以用电路图表示一下：

<img src="/img/in-post/2018-02-09-optimization2/circuit.png" width="50%">

图中绿色部分是`forward pass`计算出的值。当所有`forword pass`的值计算出来后，我们开始`back prop`。从最后的部分开始反向传播，递归地用`chain rule`来计算，直到计算出最左边的输入的梯度。 

<a name='intuitive'></a>

### Intuitive understanding of backpropagation

​	请注意，反向传播是一个非常漂亮的本地过程。电路图中的每个门都有一些输入，可以立即计算两个量：1. 其输出值 2. 其输出值相对于其输入值的`local gradient`。请注意，这些门可以完全独立地完成这个任务，而不需要知道它所嵌入的完整电路的任何细节。一旦`forward pass`结束，在`back prop`期间，通过`chain rule`相乘所有经过该门的梯度，每个门将知道整个电路的最终输出相对于该门所有输入的梯度，

>由于链规则，这种额外的乘法（对于每个输入）可以将单个相对无用的门变成复杂电路（例如整个神经网络）中的齿轮。

让我们再次参考这个例子，直观的了解这个工作原理。`add`门接收输入 $[-2,5]$ 和计算输出3。加法运算关于两个输入的`local gradient`为+1。电路的其余部分计算出最终值为-12。在反向传递过程中，`chain rule`通过电路递归地往回计算，加法门（它是乘法门的输入）获知其输出的梯度是-4。如果我们把电路拟人化用第一人称来阐述：

​	我作为整个电路想要最终有更高的输出值（当然可以是最低的输出值，这里以最高为例）。我包含有一个`add`门，由于此门的梯度为-4，所以我想要`add`门的输出更低。同时`add`门递归计算出最初的 $x,y$ 的局`local gradient`，也都是 -4。我也希望 $x, y$ 的值更低，这样`add`门的值就更低，我最终的输出值就更高。

**反向传播因此可以被认为是不同门之间的沟通（以梯度作为信号），来诉说每个门是否希望他们的输出增加或减少（以及有多强），从而使最终输出值更高（或更低）。**

<a name='sigmoid'></a>

### Modularity: Sigmoid example

我们上面介绍的门是相对任意的。 任何一种可导函数都可以作为一个门，我们可以把多个门组成一个门，或者为了计算方便，可以把一个函数分解成多个门。 让我们看一个例子：

$$
f(w,x) = \frac{1}{1+e^{-(w_0x_0 + w_1x_1 + w_2)}}
$$

这个表达式描述了使用S形激活函数的二维神经元（输入x和权重w）。但是现在让我们把它想象成一个函数，从输入w，x到单个数字。 该函数由多个门组成。 除了上面已经介绍的`add，mul，max`之外，还有四个：
$$
f(x) = \frac{1}{x} 
\hspace{1in} \rightarrow \hspace{1in} 
\frac{df}{dx} = -1/x^2 
\\\\
f_c(x) = c + x
\hspace{1in} \rightarrow \hspace{1in} 
\frac{df}{dx} = 1 
\\\\
f(x) = e^x
\hspace{1in} \rightarrow \hspace{1in} 
\frac{df}{dx} = e^x
\\\\
f_a(x) = ax
\hspace{1in} \rightarrow \hspace{1in} 
\frac{df}{dx} = a
$$
函数 $fc，fa$ 分别是 $x+c$ ，和用 $a$ 进行常数缩放。这些是特殊情况下的加法和乘法，但是我们将它们作为（新）一元门引入，因为我们确实需要常量的梯度 $c, a$ 。 完整的电路如下所示：

<img src="=/img/in-post/2018-02-09-optimization2/sigmoidCircuit.png" width="90%">

二维神经元的示例电路。 输入是$[x0，x1]$ ，神经元的学习权重是 $[w0，w1，w2]$ 。 正如我们后面将会看到的那样，神经元计算输入和权重的点积，然后将该神经元的激活阈值范围通过`sogmoid`函数压缩到0到1之内。

在上面的例子中，图示中右边横线部分的所有操作可以抽象出一个常用的函数，这个函数被称为`sigmoid`函数 $σ(x)$ 。如果我们对`sigmoid`函数求导，我们会发现很有意思的事：
$$
\sigma(x) = \frac{1}{1+e^{-x}} \\\\
\rightarrow \hspace{0.3in} \frac{d\sigma(x)}{dx} = \frac{e^{-x}}{(1+e^{-x})^2} = \left( \frac{1 + e^{-x} - 1}{1 + e^{-x}} \right) \left( \frac{1}{1+e^{-x}} \right) 
= \left( 1 - \sigma(x) \right) \sigma(x)
$$
导数非常简单！例如，`sigmoid`表达式接收输入1.0，并在正向传递期间计算输出0.73。 根据上面的推导，局部梯度为 $(1 - 0.73)* 0.73 \simeq 0.2$，和上图一步一步算出来的一样（见上图），这样一步到位计算出导数简单快速。 因此，在实际应用中，会有这些将多个门组合成单个门的操作。用python代码实现如下：

```python
w = [2, -3, -3] # assume some random weights and data
x = [-1, -2]

# forward pass
dot = w[0]*x[0] + w[1]*x[1] + w[2]
f = 1.0 / (1 + math.exp(-dot)) # sigmod function

# backward pass through the neuron (backpropagation)
ddot = (1 - f) * f # gradient on dot variable, using the sigmoid gradient derivation
dx = [w[0] * ddot, w[1] * ddot] # backprop into x
dw = [x[0] * ddot, x[1] * ddot, 1.0 * ddot] # backprop into w
# we are done! we have the gradients on the inputs to the circuit
```

**proTips**：合理分割反向传播模块。如上面代码所示，我们希望我们`forward pass`之后能够很容易进行 `back prop`。例如，这里我们创建了一个中间变量`dot`，它保存w和x的点积。之后`back prop`的时候，我们计算出`ddot`，并最终计算出`dw`和`dx`。

本节的要点是反向传播的细节以及怎样把`forward pass`函数组合成一个门能够使得计算更加方便。了解表达式哪一部分的组合导数相对简单能够帮助我们更方便地使用`chain rule`计算函数关于输入的导数。

<a name='staged'></a>

### Backprop in practice: Staged computation

举一例子看：
$$
f(x,y) = \frac{x + \sigma(y)}{\sigma(x) + (x+y)^2}
$$
这个函数没有什么特别的用处，只是一个很好的例子说明`back prop`如何工作。 强调这一点非常重要，如果你要开始对x或y进行求导，结果非常复杂。然而这样做完全没有必要，因为我们不需要写一个明确的函数来计算梯度。我们只需要知道如何计算它。 以下是我们如何构建这种表达的正向传递：

```python
x = 3 # example values
y = -4

# forward pass
sigy = 1.0 / (1 + math.exp(-y)) # sigmoid in numerator 		#(1)
num = x + sigy # numerator									#(2)
sigx = 1.0 / (1 + math.exp(-x)) # sigmoid in denominator	#(3)
xpy = x + y													#(4)
xpysqr = xpy**2												#(5)
den = sigx + xpysqr	# denominator							#(6)
invden = 1.0 / den											#(7)
f = num * invden # done!									#(8)
```

上面的一长串代码完成了`forward pass`部分。我们将代码结构化为包含多个中间变量，每个中间变量都只是我们已知道`local gradient`的简单表达式。因此，`backprop` 很简单：我们将从后往前，计算电路输出相对于每个中间变量的梯度（sigy，num，sigx，xpy，xpysqr，den，invden），并且用 `d` 前缀表示梯度。 每一步`back prop`都会计算该节点的所有输入的`local gradient`，并把他们和该函数最终输出关于该节点的梯度相乘，得到该函数关于该节点所有输入的梯度。具体看代码：（代码右边的序号对应上面代码中`forward pass`的部分）

```python
# backprop f = num * invden
dnum = invden # gradient on numerator						#(8)
dinvden = num												#(8)
# backprop invden = 1.0 / den								
dden = (-1.0 / (den**2)) * dinvden							#(7)
# backprop den = sigx + xpysqr
dsigx = (1) * dden 											#(6)
dxpysqr = (1) * dden										#(6)
# backprop xpysqr = xpy**2
dxpy = (2 * dxpy) * dxpysqr									#(5)
# backprop xpy = x + y
dx = (1) * dxpy												#(4)
dy = (1) * dxpy												#(4)
# backprop sigx = 1.0 / (1 + math.exp(-x))
dx += ((1 - sigx) * sigx) * dsigx # Notice += !! see notes below #(3)
# backprop num = x + sigy
dx += (1) * dnum											#(2)
dsigy = (1) * dnum											#(2)
# backprop sigy = 1.0 / (1 + math.exp(-y))
dy += ((1-sigy) * sigy) * dsigy								#(1)
# done ! Phew
```

注意以下几点：

**缓存 forward pass 变量：** 为了`backward pass`更快，我们会使用`forward pass`中计算出的变量。 在实践中，我们构建的代码要能够方便地缓存这些变量，使其在`back prop`期间可用。当然如果难以保存这些变量，重新计算它们也是可能的，只不过会造成算力浪费。

**梯度 += 符号的意义：**`forward pass`表达式涉及x, y 多次，所以当我们执行`back prop`时，必须使用 `+=`而不是`=`来累积这些变量的梯度（而不是覆盖它）。这遵循微积分中的多变量链规则，该规则规定，如果一个变量分支到电路的不同部分，那么流回它的梯度将逐个增加。

<a name='patterns'></a>

### Patterns in backward flow

有意思的是，在很多情况下，可以直观的解释`back prop`的梯度。例如，神经网络中最常用的三个门 `add, mul, max`，在反向传播过程中它们的行为方式都有非常简单的解释。 请看以下示例电路：

<img src="/img/in-post/2018-02-09-optimization2/threeBaseCircuit.png" width="60%">

`add`将梯度平均分配给所有输入。`max`将梯度路由到较高的输入。`mul`接受`input activations`（绿色的输入值），交换它们并乘自己的梯度得到`input`的梯度。

`add`门总是将其输出上的梯度平均分配给所有输入，而不管它们在`forward pass`期间的值如何。这是因为`add`操作的`local gradient`为+1.0，所以所有输入的梯度等于输出的梯度。比如图中`add` 门将 2.00 的梯度路由到其两个输入。

`max`门路由（route）梯度。与`add`门不同，`max`门将梯度不变地分配给其输入中的一个（在正向传递期间具有最高值的输入）。这是因为`max`门的`local gradient`最高为1.0，所有其他值为0.0。在上面的示例电路中，`max`操作将2.00的梯度路由到z变量，并且w上的梯度为零。

`mul`门的局部梯度是将输入值进行交换之后，再使用链规则乘以其输出的梯度。在上面的例子中，x上的梯度是-8.00，即-4.00 x 2.00。

关于`mul`门，我们多说一点，如果`mul`门其中一个输入非常小，另一个非常大，那么乘法门会做一些不怎么直观的事情：它会为小输入分配一个相对较大的梯度，并为大输入分配一个很小的梯度。大家还记得在线性分类器中计算分数的函数 $ w^Tx_i$ 。那么$w^T$的导数就是$x_i$，说明 $x_i$ 对 $w^T$ 的梯度大小有影响。例如，如果在预处理期间将所有输入数据示例 $xi$ 乘以1000，则权重 $w^T$上的梯度将增大1000倍，随之而来的是我们必须将学习率降低以进行平衡，不然训练很容易`overshooting`（步长太大，每次调整幅度较大）。这就是为什么预处理输入很重要（还记得之前的平均化图片吗？把图片数字化的信息$[0,256]$ 转换成 $[-127, 127]$，以及把数据`normalize`到 $[0,1]$之间，这些都是良好的预处理步骤）。

<a name='mat'></a>

### Gradients for vectorized operations

上述部分讨论的都是单个变量，现在我们把概念都直接扩展到矩阵和向量操作。 但是，我们必须更加关注维度和转置操作。

**Matrix-Matrix multiply gradient** 可能最棘手的操作是矩阵-矩阵乘法（它是所有矩阵向量乘法和向量向量乘法的总概括）：

```python
# forward pass
W = np.random.randn(5, 10)
X = np.random.randn(10, 3)
D = W.dot(X)

# now suppose we had the gradient on D from above in the circuit
dD = np.random.randn(*D.shape) # same shape as D
dW = dD.dot(X.T) #.T gives the transpose of the matrix
dX = W.T.dot(dD)
```
提示：使用维度分析！你不需要记住`dW`和`dX`的表达式，因为它们很容易根据维度重新派生。例如，我们知道权重的梯度`dW`在计算之后必须和`W`的维度相同，并且它必须依赖于`X`和`dD`的矩阵乘法，这时候我们就通过维度来分析。例如，`X`的大小为`[10 x 3]`，`dD`的大小为`[5 x 3]`，所以如果我们想要`dW`和`W`的形状一样，都为`[5 x 10]`，那么实现这一点的唯一方法是使用`dD.dot(X.T)`，如上所示。

Erik Learned-Miller还撰写了一篇关于矩阵/矢量求导的文章，或许对你有帮助，[戳这里](http://cs231n.stanford.edu/vecDerivs.pdf)。

<a name='summary'></a>

### Summary

- 我们建立了梯度的直觉感受。它们如何在电路中`back prop`以及它们如何沟通电路的哪一部分应该增加或减少，以及如何使最终输出更高。
- 我们讨论了分模块计算对`back prop`实现的重要性。你总是希望将你的函数分解成特定模块，你可以很容易地求出`local gradient`，然后用`chain rule`链接它们。最重要的是，我们不需要一次性求整个表达式的复杂导数。因此，将表达式分解为多个阶段，以便可以独立求导每个模块（比如矩阵向量相乘，或者最大操作或求和操作等），然后一步一步地反向传递变量。

在下一节中，我们将开始定义神经网络，`backprop`会帮助我们快速地计算神经网络中损失函数的梯度。换句话说，我们现在已经准备好训练神经网络了，我们已经把课程中最难的部分搞定啦，而之后的`ConvNets`只是再前进一小步。

版权声明：

- 自由转载-非商用-非衍生-保持署名（[创意共享3.0许可证](http://creativecommons.org/licenses/by-nc-nd/3.0/deed.zh)）
---
layout: post
title: "cs231n Lecture 5<br>Convolutional Neural Networks Notes<br>卷积神经网络笔记"
date: 2018-02-16
author: "injuryholmes"
catalog: false
header-img: "img/post-bg-cs231n.jpg"
tags:
    - convolutional neural networks
---

cs231N是斯坦福大学的卷积神经网络课程，我通过博客记录自己的学习心得，前几篇博文的翻译工作不仅工作量大，而且由于水平有限，故很多地方不明所以，这篇博文改了纯翻译的形式，更多的是挑一些点阐述。推荐读者阅读完英文原文之后再来阅读。

*本博客仅做参考，如有措辞不当或理解错误，欢迎大家留言指正。*

更多原文内容请查看[官网](http://cs231n.github.io/convolutional-networks/)。

- [关键词一览](#keywords)
- [一张图解释卷积层 CONV layer](#convdef)
  - [我的错误理解](#wrongdef)
  - [我的正确理解](#correctdef)
- [如何理解 FC layer 和 CONV layer 的转化](#fc2conv)
- [堆叠CONV layer的好处](#multiconv)

<a name='keywords'></a>

## 关键词一览：

- `downsample`：降采样 

> **本文中侧重的是降低数据大小**
>
> 在数字信号处理领域中，降采样是一种多速率数字信号处理的技术或是降低信号采样率的过程，通常用于降低数据传输速率或者数据大小。 

- `input volume` ：输入体积，侧重`volume`
- `activation volume`：有效（输入、输出）体积，侧重有效和空间体积
- `receptive field`: 感受野，也就是 `filter size`
- `depth column`: 深度列
- `depth slice`：深度切片
- `neuron`：神经元
- `filter`：过滤器
- `activation map`: 单层有效（输出）体积，侧重depth slice
- `non-linearity`：非线性函数，比如ReLu函数，往往跟在`CONV layer`之后

<a name='convdef'></a>

## 一张图解释卷积层 CONV layer

1. 区分`CONV layer`**参数**和`CONV layer output volume`

   <img src="/img/in-post/2018-02-16-cnn/convdef.png">

   首先，看着这张图，在蓝色的长方体中，问自己三个问题：

   - 一个球代表什么？
   - 一列`depth column`里所有的球代表什么？
   - 一片`depth slice`里所有的球代表什么？

<a name="wrongdef"></a>

### 一个球代表什么？

一个球就是一个`neuron`，这个`neuron`代表的是一种`filter`，用来检测**某一种**特征，比如横线或竖线或对角线等等。

### 一列`depth column`里所有的球代表什么？

一个球就是一个`neuron`，代表的是一种`filter`，那么在`depth column`中的一串球，就是`neurons`，每一个`filter`检测一种特征，比如在这里从左至右依次检测横线、竖线、左上至右下对角线、右上至左下对角线等等。

### 一片`depth slice`里所有的球代表什么？

请先阅读原文关于`parameter sharing`部分。

原文中有一个假设：

> 如果一个特征参数对于某个空间位置 $(x,y)$ 有用，那么在另一个位置 $(x2,y2)$ 上可能会是有用的。在卷积层中，一个过滤器包含有一个特征参数。只要某个过滤器扫描到符合的特征就会被激活。

也就是说，球A在扫描`input volume`的时候，只要扫描到横线，就被激活了。那么为了记录这个横线在原`input volume`中的位置，我们把激活的地点记录下来作为`filter A`的`activation map`，也就是一个`depth slice`（即蓝色长方体中的一个切面），这样`depth`个数的`filter`就对应`depth`个数的`activation map`，我们把他们堆叠在一起，就是我们的图中蓝色的长方体。

**仔细想想是这么一回事吗?**其实上面我说的都是错的，如果你也是这样想的话，就和我一开始犯了同样的错误。

<a name="correctdef"></a>

**接下来说正确的理解**

我们一开始不是说球A是`filter`吗，那怎么一个切面的`filters`还用来表示`activation map`了呢？

所以，这个图实际上可以被拆分为如下图：

<img src="/img/in-post/2018-02-16-cnn/convlayer.jpeg">

可以看到，每一个`filter` 都对应一个`depth slice`，图中用阴影部分代表的是`filter A`对应的`depth slice`，这样，这样`depth slice`是该`filter`和`input volume`的`activation map`也就不难理解了。

我啰嗦一点纠正一下自己的答案：

<img src="/img/in-post/2018-02-16-cnn/convdef.png">

### 一个球代表什么？

一个球代表`input volume`对应的某一个`local region`对于这个球所属的`filter`的激活能力值。

### 一列`depth column`里所有的球代表什么？

`depth column`中的一串球，表示的是`input volume`对应的某一个`local region`所有不同`filter`依次的激活能力值。

### 一片`depth slice`里所有的球代表什么？

`depth slice`中的所有球表示的是整个`input volume`各个不同`region`对于某一个`filter`的激活能力值。

### 整个长方体里的所有球代表什么?

就是你们在原文中常常看到的`output volume`。



那么问题来了



**我们平时说的`CONV layer`中存储的到底是`filters`还是`activation maps`？**

我们平时所说的`CONV layer`当然是`filters`啦！因为我们最后需要训练得出的模型，本质上是一个函数，这个函数能够用在不同的场景中处理图片识别的任务，所以我们需要记录的是`filters`的参数而不是`activation maps`，但是在训练的过程中，我们仍然要记录`activation maps`的值，以便之后进行`back prop`对`gradient`进行计算。

<a name="fc2conv"></a>

## 如何理解 FC layer 和 CONV layer 的转化

在理解这个之前，我们先看一下一个卷积神经网络的结构。可以用以下这个公式表示：

`INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC`

​	如果要再抠细节的话，按照上文所说，CNN的本质是一个函数，所以不应当`INPUT`放到CNN的结构中。这里暂不讨论这个细节。

​	在CNN中，有一个概念叫做`translational invariance`。我用一个例子来解释：

​	正如我们上面看到CNN的经典结构，比如，我们有10个`CONV layers`，中间夹着必要的`POOL layers`，神经网络的最后一层使用`FC layer`点对点连接通过之前10个`CONV layers`提取出来的特征信息。`LeNet`就是这样的模型。但是，这样的模型有一个问题，就是它只能针对特定大小的图片输入，假设我们的输入图片的尺寸是 $28*28$，经过一连串卷积和下采样之后，得到尺寸为 $12*12$ 的特征信息，如果我们用`FC layer`去连接这一层，我们就必须要构建含有 $12*12=144$ 个神经元的`FC layer`，这在图像检测（`object detection`）的实际使用中，是非常不方便的，因为我们的输入图像往往不是一样大的。但是把`FC layer`转化成`CONV layer`就方便多了。原文中举了这么个例子：（摘自原文）

<div>

<p>For example, if $224*224$ image gives a volume of size $7*7*512$ - i.e. a reduction by 32, then forwarding an image of size $384*384$ through the converted architecture would give the equivalent volume in size $[12*12*512]$, since $384/32 = 12$. Following through with the next 3 CONV layers that we just converted from FC layers would now give the final volume of size $[6*6*1000]$, since $(12 - 7)/1 + 1 = 6$. Note that instead of a single vector of class scores of size $[1*1*1000]$, we’re now getting an entire $6*6$ array of class scores across the $384*384$ image.
valuating the original ConvNet (with FC layers) independently across $224*224$ crops of the $384*384$ image in strides of 32 pixels gives an identical result to forwarding the converted ConvNet one time.
aturally, forwarding the converted ConvNet a single time is much more efficient than iterating the original ConvNet over all those 36 locations, since the 36 evaluations share computation. This trick is often used in practice to get better performance, where for example, it is common to resize an image to make it bigger, use a converted ConvNet to evaluate the class scores at many spatial positions and then average the class scores.

</p>

​	也就是说原本输入是$224*224$，经过卷积和下采样后得到大小为 $7*7*512$ 的输出，如果接着使用`FC layer`，我们就需要有 $7*7*512=25088$ 个神经元，而且当我们的输入图片变成 $384*384$ 之后，之前训练得到的`FC layer` 的数据都没有用了，因为我们需要更大的，含有 $12*12*512 = 73728$个神经元的`FC layer`，显然，这不OK。

​	但是如果把最后的`FC layer`转化成`CONV layer`会怎么样呢？以原文中的为例，这种情况下，当输入图片的大小变成 $384*384$ 的时候，我们最后得到的是 $[6*6*1000]$ 而不是 $[1*1*1000]$，但是我们可以把这 $6*6$ 的方格当成一个整体，原来用一个值代表某个特征可能性的大小，现在用一个方格表示某个特征可能性的大小，当然把方格看成整体的数学意义在这里不做深入讨论，笔者也没有深入了解，就不误导读者了。

<a name="multiconv"></a>

## 堆叠CONV layer的好处

<p>原文中的意思就是，使用 $3*3$ 的`CONV layer`，在`layer3`中的一个神经元，向上延伸至`input volume`，可以连接到7个神经元，也就是说，在第三层中，每个神经元能够提取的信息是基于原始输入中 $[7*7]$ 的局部区域。</p>

<img src="/img/in-post/2018-02-16-cnn/muilticonv.jpeg"> 

<img src="/img/in-post/2018-02-16-cnn/relu.png" width="50%">

根据示意图，由于每一层`CONV layer`之后都会紧接着跟着一个`non-linearty`，这里以`ReLu`为例（附上Relu函数的图片），所以，按照图一的方式，我们对原图特征的把控更加细腻，注重不同细节。如果把三层 $[3*3]$ 的`CONV layer`改成1层 $7*7$ 的`CONV layer`，对应的就只会使用一次`Relu`函数，对于原图的细节把握没有三层的到位，单层的`ReLu`也会导致一些不明显的特征被忽略。



希望这篇博文对你有帮助。



版权声明：

- 自由转载-非商用-非衍生-保持署名（[创意共享4.0许可证](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.zh)）




---
layout: post
title: "Image Classification Notes<br>图片分类器笔记"
date: 2018-01-18
author: "injuryholmes"
catalog: false
tags:
    - Image Classification
    - Nearest Neighbor Classifier
---

cs231N是斯坦福大学的卷积神经网络课程，我通过博客记录自己的学习心得，对于原文中一些难点做了一下详细的解释。希望能给看文章的你带来帮助。

*为了方便大家对照原文，我尽量在直白翻译和自己的理解之间做权衡，所以文章读起来会有些琐碎。*

*本博客仅做参考，如有措辞不当或理解错误，欢迎大家留言指正。*

更多原文内容请查看[官网](http://cs231n.stanford.edu/)。

- [图片分类器介绍(Intro to Image Classification)，数据驱动方法(Data-driven approach)，流水线(Pipeline)](#intro)
- [最近邻居法 (Nearest Neighbor Classifier)](#nn)
  - [K-最近邻居法(K-Nearest Neighbor)](#knn)
- [验证集(Validation sets)，交叉验证(Cross-validation)](#val)
- [NN算法的优缺点(Pros/Cons of NN)](#procon)
- [总结](#summary)
- [总结: 如何使用 kNN](#summaryapply)


<a name="intro"></a>

### Intro to Image Classification

- 定义： 

  图片分类器能根据待测试图片的信息，从已有的标签集合中选取合适的标签。

- 举例：

  ​	假设我们有标签集合 {猫，狗，帽子，马克杯}。下图为待测试图片，计算机能够根据我们写的算法，给该图片贴上“猫”的标签。人能直观的看到一只猫，但是计算机看到的是一个三维的数组，因为一个像素(pixel)是由R(red)、G(green)、B(blue)三个子像素构成。这张猫的图片是248宽 * 400高。所以将这个图片 __*数字化*__ 之后，总共包含248 * 400 * 3 = 297600个数字信息，存储在一个三维数组中。每一个数字都是一个从 0(黑色) 到 255(白色) 之间。

  <img src="/img/in-post/2018-01-18-image-classification-notes/imageClassification_cat.png" alt="mageClassification_cat" width="75%">


- 小贴士：

  - 在之后的文章中，你会发现，在计算机视觉领域，比如物体探测（object detection）、分割（segmentation）等等这些任务都能简化成图片识别。

- 图片识别中需要克服的困难：

  - 不同的视角（viewpoint variation）: 一张猫咪的照片可以是从不同的角度被拍摄的。

  - 不同的大小（size variation）: 同一个猫咪在照片里可以被拍的大一点或者小一点，不仅如此，猫咪品种很多，体型自然有大有小。

  - 变形（deformation）: 猫咪往往具有一定的延展性。

    <img src="/img/in-post/2018-01-18-image-classification-notes/deformation.png" alt="deformation" width="25%">

  - 遮挡（occlusion）: 猫咪的一部分被遮挡，只有部分的猫咪能陪看到。

    <img src="/img/in-post/2018-01-18-image-classification-notes/occlusion.png" alt="occlusion" width="25%">

  - 光照条件（illumination conditions）: 因为图片数字化的本质是像素，所以照明条件对图片有很大的影响。

    <img src="/img/in-post/2018-01-18-image-classification-notes/illumination.png" alt="illumination" width="25%">

  - 混杂的背景（background clutter）: 猫咪和背景融为一体，导致图片区分度下降。
    <img src="/img/in-post/2018-01-18-image-classification-notes/bgClutter.png" alt="background clutter" width="25%">

  - 繁多的种类（intra-class variation）: 以下图右下角的椅子为例子，不同的椅子有自己不同的外形。

<img src="/img/in-post/2018-01-18-image-classification-notes/challenges.png" alt="challenges">

​	以上列出了一些常见的问题，一个好的图片分类器模型应当能够不管以上问题以何种组合出现时，都不受到影响。同时能够保持对繁多种类（intra-class variation）的敏感度。

**大数据驱动法（Data-driven approach）：**

​	我们通过大量地喂给计算机不同分类的已标记图片，然后构建一个基于这些大数据的算法，来对每个类进行识别。就像我们教一个新生儿认识一棵树一个道理。这种名为大数据驱动法基于大量的已知数据（也叫训练集training set），下面是一个训练集的例子。

<img src="/img/in-post/2018-01-18-image-classification-notes/trainingSet.png" alt="training Set" width="90%">

- 小贴士：
  - 实际情况中，训练数据中每个类都有成千上万的已标记图片。

**图片分类器流水线（Pipeline）：**

​	从上面猫咪的例子可以看到，图片分类器就通过对像素数组的分析，给这个数组贴上一个标签。我们可以把这个过程分为以下步骤：
1. 输入（Input）：输入是已标记的图片的集合。我们把这些图片叫做“训练集” （training set）。
2. 学习（Learning）: 利用之前传入的训练集来训练一个分类器（classifier），也可以叫做模型（model）。
3. 评估（Evaluation）: 我们要对步骤2中训练出来的分类器进行评估。将分类器预测图片的标签和图片真实的标签（也称作 ground truth）进行对比。计算分类器的准确率。

<a name="nn"></a>

### Nearest Neighbor Classifier

​	我们介绍的第一个图片分类器叫做“最近邻居法”。这个分类器的思想和卷及神经网络(Convolutional Neural Networks)基本没有关系，并且在实际生产中的使用也相当少。但是从这个简单的例子中，有助于加深我们对图片识别的认识。

**样例图片集：[CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)**

​	CIFAR-10拥有60000张图片，分为10个类别。每张图片的像素是32*32，并且都有自己的标签，这6000张图片被分为两个部分，50000张图片作为训练集（training set），剩下10000张图片作为测试集（test set）。下面的例子，分别从每个类型随机挑选了10张图片。

<img src="/img/in-post/2018-01-18-image-classification-notes/cifar_10.png" alt="cifar_10">

​	上图中，左边的样本取自CIFAR-10，右边的图片第一列就是我们的待测试图片，然后通过“最近邻居法”，算出在训练集中，离这些测试图片最接近的10张图片。

​	那么问题来了，我们凭什么说一张图片和另一张图片离的比较近呢？

接下来，就要引出两个距离方程。

##### L1(曼哈顿城市距离)

> $$d_1(I_1, I_2) = \sum_p{|I_1^p - I_2^p|}$$

​	I是Image的缩写，p上标是对不同像素(pixel)的索引。所以用普通话说，这个L1距离，就是两张图片对应像素点之间的绝对值之和。这个绝对值越小，说明两张图片每个对应像素点越接近，这两张图片属于一个标签的可能性越大；同理，这个绝对值越大，说明两张图片对应像素点差的越远，这两张图片属于一个标签的可能性越小。

<img src="/img/in-post/2018-01-18-image-classification-notes/L1distance.png" alt="L1distance">

##### L2(欧几里得距离)

> $$d_2(I_1, I_2) = \sqrt{\sum_p{(I_1^p - I_2^p)}^2}$$

​	在L2距离的计算中，我们求的是对应像素点差的平方和，再求其算术平方根。值得一提的一点是，在实际的代码使用中，往往我们不会做最后一步的开方，因为开方对最后结果的比较没有影响（像开方这样，拥有不改变顺序性质的函数，我们称之为monotonic function）。开方只改变了所有计算出的距离值的绝对大小，但是没有改变这些距离们的顺序。

​	好，知道了理论，来看一下代码实现，这里以L1距离为例。

- 在正式代码开始之前，给大家提一个注意点，关于什么是一维数组

```python
>>> d1 = np.array([0, 0, 0])
>>> d1
array([0, 0, 0])
>>> d1.shape
(3,)
>>> d2 = np.array([[0,0,0]])
>>> d2
array([[0, 0, 0]])
>>> d2.shape
(1, 3)
```

​	这里希望大家明白`(1,3)`其实是二维数组，不要因为二维数组只有一行，就误认为是一维数组，而在numpy中一维数组的维度表示形式是`(length, )`。

​	这里我们就认为`load_CIFAR10()` 是提供给你的API，你不管这个函数怎么实现的，你只要知道，这个函数内部是将50000张图片作为训练集，10000张图片作为测试集，并返回了以下几个变量。

- Xtr: 所有训练集图片数字化之后的三维数组，Xtr维度是`(50000, 32, 32, 3)`。
- Ytr: 所有训练集图片对应的标签，Ytr的维度是`(50000, )`。就是大家最熟悉的一维数组，每个数组的元素是一个数字（0-9代表不同的类）。
- Xte: 所有测试集图片数字化之后的三位数组，维度 `(10000, 32, 32, 3)`。
- Yte: 所有测试集图片对应的标签，维度`(10000, )`。

```python
Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar10/') # a magic function we provide
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows is (50000, 3072)
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows is (10000, 3072)
```

​	`reshape()`的作用就像一把大锤子，嘭嘭一顿敲，原来`(50000, 32, 32, 3)`的高维数组，变成了 `(50000, 3072)`的数组，达到了降维的效果，这样，又变成了大家熟悉的二维数组。总共50000行，3072列，每一行表示一张图片的数字化信息。我们这么做是因为操作更加的简单和直观。

```python
nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print 'accuracy: %f' % ( np.mean(Yte_predict == Yte) )
```

​	我们常常使用准确率来衡量模型的好坏。注意，上面的`train(X, y)` 函数接收两个参数，X代表数字化并降维后的训练集，y代表对应的标签。`predict(Xte_rows)` 函数接收数字化并降维后的测试集图片为参数，返回的是模型预测的结果`Yte_predict`, 最后和`Yte` 对比求得准确率。

​	最后，附上`NearestNeighbor`的代码实现：

```python
import numpy as np

class NearsetNeighbor(object):
    def __init__(self):
    	pass
    
    def train(self, X, y):
		""" X是(N, D)的二维数组，每一行表示一张数字化并降维后的图片, Y是一维数组，(N, )"""
		# the nearest neighbor classifier simply remembers all the training data
		self.Xtr = X
		self.ytr = y
       
	def predict(self, X):
        """ X 是待测试图片的集合，也是一个数字化并降维后的二维数组 """
        num_test = X.shape[0] # 获得行数，也就是需要预测的图片总数量
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype) # 以num_test行为行数，以self.ytr.dtype作为数据类型，创建一个一维数组，用来存预测的标签值。
        # loop through all the pics from the test set
        for i in xrange(num_test):
            # 使用L1距离公式，从已存储的训练集中找到距离第i个待测试图片距离最近的图片。
            # X[i, :]表示测试集中第i行表示的图片的数字信息，提取出来之后，是一个一维的数组，维度是(3072, )，而不是(1, 3072)
            # 这里注意一点，self.Xtr是二维的，本例子中是(50000, 3072)， 那么一个二维的数组是怎么和一维的数组(3072, )相减的呢？这里就用到了numpy的broadcasting语法糖，允许我们对不同维度的数组进行运算。具体请看下一块代码。
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            # np.abs(self.Xtr - X[i, :])返回的二维数组维度(50000, 3072)
            # np.sum(二维数组, axis = 1)这里的axis表示按照哪一个维度进行求和，这里dim0=50000, dim1=3072, 即按照dim1进行求和，最后的distancecs的维度是(50000, )
            # argmin获得最小距离值对应的下标即训练集中距离待测图片最近的一张。
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]
        return Ypred
        
```

Numpy中Broadcasting语法糖

```python
>>> x = np.array([[1,2,3],[4,5,6]],np.int32)
>>> y = np.array([[1,2,3]],np.int32)
>>> x.shape
(2, 3)
>>> y.shape
(1, 3)
>>> x-y
array([[0, 0, 0],
       [3, 3, 3]], dtype=int32)
```

- 更多详情请点击[Python的numpy的Broadcasting语法糖](http://cs231n.github.io/python-numpy-tutorial/#numpy-broadcasting)
- 大家对于`sum(X, axis=)` 中参数不了解的，可以点击[这里](https://stackoverflow.com/questions/40857930/how-does-numpy-sum-with-axis-work)。

  ​这段代码跑出来的结果是在CIFAR-10上的预测准确率在 __38.6%__ 左右。这么简单的算法已经比随机瞎猜准很多了，瞎猜的概率大概是 __10%__ （因为总共有10类）。不过还是远远不及人类的识别力（__94%__），卷积神经网络目前能达到的准确率是 __95%__，可以在kaggle上[查看](https://www.kaggle.com/c/cifar-10/leaderboard) 。

  ​如果使用L2距离，只需改一行代码即可。

```python
distance = np.sqrt(np.sum(np.square(self.Xtr - X[i, :]), axis = 1))
```

​	我们在这里调用了`np.sqrt`， 但在实际应用中，我们往往不开方，因为这个函数拥有 *monotonic* 性质，它只改变距离数组中每个值的绝对大小，但是并不会影响数组的排序。所以对最近邻居的判定没有什么影响。使用L2距离获得的准确率在 __35.4%__ 左右。

##### L1 vs L2 如何选择距离公式？

​	L2距离公式比L1距离公式严格，或者说更加大惊小怪一点，L2距离公式能够容忍一点小的不同，但是两张图片进行比较的时候，一些局部像素点的巨大差距就会对两张图片相似度产生很大的影响。L1和L2距离函数是最常用的两种距离公式（或者是用L1-norm或L2-norm，效果是一样的）。

<a name="knn"></a>

### K-Nearest Neighbor Classifier

​	由于只根据最近的一个图片来确定标签存在的偶然性比较大，所以扩展一下Nearest Neighbor Classifier成K-Nearest Neighbor Classifier。这一次，我们从训练集（trainning set）中找到前k个与待测图片距离最小的已标记图片。

所以，Nearest Neighbor Classifier 就是 K = 1的特殊情况。

让我们对比K = 5和K = 1两个图片分类器的区别。

<img src="/img/in-post/2018-01-18-image-classification-notes/kneighbor.png" alt="knn">

​	我们的图片中包含三种颜色的点（红、绿、蓝），本例中，我们使用L2距离公式。通过颜色区块来表示图片分类器预测的边界（decision boundaries）。白色的区域表示分类器同时给这片区域的点两种颜色以上的标签预测。我们可以注意到，在NN分类器中，离群的点（outlier）（比如，在中间蓝色区域当中的绿色点就是一个离群的点），显然这个绿色的点不应该在蓝色的区域中划分出一块绿色的区域。而用5NN，就提高了一定的容错率，能更好地概括（generalization）所有的测试数据。

在实际的生产中，我们基本都会使用K > 1的值，但是究竟如何确定K的值呢？

<a name='val'></a>

### 使用验证集调整超参数

​	k-最近邻居分类器需要对k进行设置。但是，什么数字最适合？另外，我们看到有许多不同的距离函数可供我们使用：L1-norm，L2-norm，还有许多我们甚至没有考虑过的其他选择（例如点积）。这些选择被称为超参数，它们经常出现在许多从数据中学习的机器学习算法中。我们应当如何选择？

​	你可能会建议我们尝试许多不同的超参数值，看看什么效果最好。这确实是我们要做的事情，但记住一点，**我们不能使用测试集来调整超参数！！！**。无论何时设计机器学习算法，都应该将测试集作为一个非常宝贵的资源，理想情况下，直到最后一次之前，永远不会被使用。否则，最后在测试集上超常工作的模型，应用到实际问题中，性能可能会显著下降。这叫对过度适应（__overfit__）测试集。

​	幸运的是，有办法在不碰测试集的情况下调整超参数。我们的想法是把我们的训练集分成两部分：一个稍小的训练集，以及我们所说的验证集。 以CIFAR-10为例，我们可以使用49,000个训练图像进行训练，并留出1,000个用于验证。 该验证集本质上用作"假测试集"来调整超参数。

以下是CIFAR-10的情况：

- 训练集（trainnign set） 49,000
- 验证集 （validation set） 1,000
- 测试集 （test set） 10,000

将你的训练集分成训练集和验证集。 使用验证集来调整所有超参数。 最后在测试集上运行一次并报告性能。

```python
# assume we have Xtr_rows, Ytr, Xte_rows, Yte as before
# recall Xtr_rows is 50,000 x 3072 matrix
Xval_rows = Xtr_rows[:1000, :] # take first 1000 for validation
Yval = Ytr[:1000]
Xtr_rows = Xtr_rows[1000:, :] # keep last 49,000 for train
Ytr = Ytr[1000:]

# find hyperparameters that work best on the validation set
validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:
  
  # use a particular value of k and evaluation on validation data
  nn = NearestNeighbor()
  nn.train(Xtr_rows, Ytr)
  # here we assume a modified NearestNeighbor class that can take a k as input
  Yval_predict = nn.predict(Xval_rows, k = k)
  acc = np.mean(Yval_predict == Yval)
  print 'accuracy: %f' % (acc,)

  # keep track of what works on the validation set
  validation_accuracies.append((k, acc))
```

​	在这个过程的最后，我们可以绘制一个图表，显示哪个k值最好。 然后，用这个值，在实际测试集上评估一次。

**交叉验证 (cross-validation)**

​	如果训练数据比较少，（同时验证集也小），那么我们可以使用更复杂的超参数调整技术，称为交叉验证。承接上面的例子，不是任意选取前1000个数据点作为验证集合和其余的训练集合，而是可以通过多次选择不同的验证集和训练集以平均性能。 例如，在5倍交叉验证（5-fold-cross-validation）中，我们将训练数据分成5个相等的组，使用其中4个组用于训练，1个组用于验证。 然后，我们将迭代验证组的倍数（5次），评估性能，最终得到一个相对平均的超参数值。

<img src="/img/in-post/2018-01-18-image-classification-notes/crossValidation.png" alt="crossValidation">

​	我们使用5倍交叉验证（5-fold-cross-validation）。不管K值取多少，我们都会进行5次的验证。上表中，y-轴是模型预测的准确率，x轴是K值。我们可以看到，每个x值对应的y值有5个，分别对应选择不同训练集和验证集组合得到的模型准确率。我们取这5个准确率的平均值，构成折线图。折线图表示，当K = 7的时候，模型拥有更好的预测准确度。如果我们使用更高倍的交叉验证法，我们将会得到更加平滑的折线图。

​	然而在实际的生产中，我们更多的用一个验证集而不是用交叉验证。因为多次的迭代训练模型，是需要消耗巨大的算力的。如果总的数据比较少，我们为了避免偶然性，还是会选择交叉验证。这个时候，我们把训练集中50%-90%的数据用于训练模型，剩下的用于验证。比例的多少取决于很多因素，比如，如果超参数比较多，那么我们倾向于更大的验证集，超参数比较少的时候，验证集可以相对小一些。

<a name='procon'></a>

**最近邻居法优缺点分析：**

优点：

- 算法的逻辑非常简单，代码的实现也不复杂。
- 分类器的训练效率很高，我们所要做的就是把训练集存储起来。

缺点：

- 分类器在测试图片的时候，需要和训练集中大量的已标记图片一一比较，因此耗费大量的算力和时间。在实际生产中，相比于训练时的效率，我们跟注重测试图片时的效率。

拓展：

- 深度神经网络（Deep Neural networks） 扭转了这种“训练-测试”的效率模式，使得训练一个神经网络变得非常的昂贵，但是，一旦训练结束，测试环节就变得非常的简单快速。这更像是实际生产中需要的模型。

- 算法科学家们研究并尝试优化最近邻居法的算法复杂度，比如，许多基于ANN（Approximate Nearest Neighbor）的算法，可以加速在训练集中的查询时间（比如，<a href="http://www.cs.ubc.ca/research/flann/">FLANN算法</a>）。这一类估计算法在训练集中查找邻居图片的时候，通过牺牲一定的准确率来优化空间/时间复杂度，这类算法往往需要预处理，例如创建一个kd树，或者使用k-means的算法。

- 尽管有这些优化的算法，在实际应用中，最近邻居法仍然很少用于图片识别。因为“图片”是高维物体（图片包含非常多的像素），在高维空间中，距离的概念往往不是那么直观，所以会导致一种情况：一张图片和好多不同的图片距离都相等，但是这些图片本身却不尽相同。

  <img src="/img/in-post/2018-01-18-image-classification-notes/unituitive.png" alt="unituitive">

  通过距离公式分别计算上述图片original和shifted（移动了一点位置），messed up，还有darkened三者的距离。将三个距离记作$d_1$, $d_2$, $d_3$, 尽管这里$d_1 == d_2 == d_3$ (我们人为构图使得后三张图的像素点满足$d_1 == d_2 == d_3$)，但是这并不代表后面三张图片本身是一样的。

  ​	这里还有一个例子，说明光使用像素差异来比较图像是不够的。 我们可以使用称为[t-SNE](http://homepage.tudelft.nl/19j49/t-SNE.html)的可视化技术来获取CIFAR-10图像，并将它们嵌入到两个维度中，以便最好地保留它们（局部）的成对距离。 在这个可视化中，使用的是L2距离，附近显示的图像被认为是非常接近的。

  大图戳[这里](http://cs231n.github.io/assets/pixels_embed_cifar10_big.jpg)。

  ​	尤其要注意的是，彼此相邻的图像更多地是图像的一般颜色分布或背景类型相符合了，而不是其语义上的含义。 例如，一只狗可以看到非常接近一只青蛙，因为它们都发生在白色背景上。 理想情况下，我们希望所有10个类的图像形成自己的聚类，使得同一类的图像彼此相邻。

<a name='summary'></a>

### 总结：

- 我们介绍了__图片分类（Image Classfication）__的问题，在这个问题中，我们给出了一组图片，这些图片都被标记了一个类别。然后我们需要根据这些已知数据预测待测试图片的类别，并测量预测的准确度。
- 我们引入了简单的分类器——"最近邻居法"分类器。模型中有多个超参数（例如k值，距离函数的选择），这些超参数的选择没有一个很显而易见的方法。
- 设置这些超参数的正确方法是将训练数据分成两部分：训练集和验证集，我们尝试不同的超参数值，最后选择模型准确率最高的超参数组合。
- 如果训练数据比较小，我们会使用交叉验证的方法，能够降低噪点带来的影响，帮我们选择更好的超参数组合。
- 一旦最好的超参数组合确定后，我们在实际测试集中，只运行一次。
- 我们看到，最近邻居法在CIFAR-10的识别准确率在40%左右。这个算法实现简单，但是需要我们存储大量的训练集，并且在预测图片的时候，耗费很长的算力和时间。
- 最后，我们看到在原始像素上直接使用$L_1$和$L_2$是不够的，因为这样会使分类器更依赖于图片的背景、像素分布而不是图片中内容在语言学上的真正表示的意义。


<a name='summaryapply'></a>

### 总结：如何在实际生产中使用KNN

- 对数据进行预处理：将数据中的要素（例如图片中的一个像素）标准化为零均值（zero mean）和单位差异（unit variance）。我们将在之后的文章中更详细的介绍相关内容。我们在本文中没有使用数据规范化（data normalization），是因为图片中的像素通常是均匀的，并且没有表现出广泛的不同的分布。所以就减少了数据规范化的需要。
- 如果你的数据是非常高维的，可以考虑降维技术，比如PCA（ ([wiki ref](http://en.wikipedia.org/wiki/Principal_component_analysis), [CS229ref](http://cs229.stanford.edu/notes/cs229-notes10.pdf), [blog ref](http://www.bigdataexaminer.com/understanding-dimensionality-reduction-principal-component-analysis-and-singular-value-decomposition/)) or even [Random Projections](http://scikit-learn.org/stable/modules/random_projection.html) ）
- 在对训练数据分为训练集和验证集的时候，按照经验，70%-90%的数据划分给训练集。这个划分的比例取决于你有多少超参数以及你期望他们有多少影响力。如果有多个超参数需要确定，那么更大的验证集能够更高效准确的确定超参数的值。但如果你不希望验证集太大，或者本身训练数据就非常少，那么最好将训练数据拆分为不同段，并执行交叉验证，如果你是土豪，能承担巨大的计算消耗，那么交叉验证吧！因为它能给出一个更加稳定准确的超参数组合。
- 训练KNN的时候，在进行验证集调整超参数的时候，尝试不同的K值，不同的距离公式。想我们上文中的折线图一样，更多的尝试能够得到更好的超参数组合，但是算力的消耗就越大。
- 如果你的KNN分类器太慢了，那么你可以考虑使用ANN（Approximate Nearest Neighbor），比如[FLANN](http://www.cs.ubc.ca/research/flann/) ，通过牺牲一定的准确率来提升分类速度。
- 记录不同超参数组合的预测准确率。这里要注意的是，如果将验证集也并入训练集，也就是将整个训练数据都作为训练集，我们可能获得更加优化的超参数集合（因为训练集的数据更大了，噪点影响就相对小了）。但是，在实际生产中，我们只希望验证集在用来调整超参数，而不要把验证集本身也放到训练数据中去。最后验证完确定超参数后，将模型在测试集上运行一次，记录在测试集上的预测准确率。
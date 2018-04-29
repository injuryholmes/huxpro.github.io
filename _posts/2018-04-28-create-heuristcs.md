---
layout: post
title: "Simulated annealing and Greate deluge<br>退火算法和大洪水算法有感"
date: 2018-04-28
author: "injuryholmes"
catalog: false
header-img: "img/in-post/2018-4-27-create-heuristics/bg.jpg"
tags:
    - Simulated annealing
    - Greate deluge
---

​	最近复习考试，其中有一门课是Artificial Intelligence methods. 我发现一个很有趣的事，分享一下

​	我发现Great Deluge和simulated annealing算法本质上一模一样。这类被称作heuristics的算法，希望模拟出智能，来解决现实生活中的问题，比如High school timeable啊，Traveling Sales man Problem(TSP)等等。

​	这类问题往往是NP-hard的。所以不可能枚举。那就存在陷入local optima（局部最优解）的可能性。然后算法学家发现在迭代早期接受一些non-improve的解，能够跳出local optima，这两种算法采用的核心思想是一开始接受non-improve的解相对容易，越往后越难。

​	Greate Deluge中文“大洪水”，大致意思是一开始水位低，新的解的“好坏程度”只要大于“水位”，就接受，但此时，新的这个解不一定要比上个解更优，这样，如果上一个解是局部最优解，我们就能避开它。然后水位逐渐升高，接受不优的解也越来越难。

​	Simulated annealing中文“退火”，火势大的时候，比较容易接受non-improve的解，然后火势越来越小，不容易接受不优化的解。

​	两个算法都人为地拉长了找到最优解的时间，从而避免了搜索域的狭隘。最终增大找到全局最优解的可能性。

​	有趣的是，你发现他们说的是一个概念。随便想想，就能创造一个“吃饱算法”，人饥饿指数高的时候什么解都吃，吃了一会，饥饿指数低了，就只吃更优的解了。还有“心累算法”，谈恋爱一开始，对方多大的任性都能包容，时间久了，趋于平淡，我对你的一点点小脾气就不能忍受了。所以，如果这样两个算法也叫不同的话，自然界很多会改变的事物，都能根据它创造成一个新的算法，赋予其名称，而会改变的事物，实在是太多了。



版权声明：

- 自由转载-非商用-非衍生-保持署名（[创意共享4.0许可证](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.zh)）


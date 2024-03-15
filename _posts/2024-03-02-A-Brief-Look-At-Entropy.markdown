---
layout: post
title: "A brief look at Entropy"
date: 2024-03-02
categories: AI
---

# Introduction

Lets take a breif look at entropy and it's relevance in machine learning models. Entropy frequently appears within machine learning because the concept is utilized to trained machine learning models. For example, cross entropy is one of the most common loss function in machine learning. 

## Entropy

Entropy in the context of information theory was introduced by Claude Shannon when he was a graduate student (Insane!). Shannon defined entropy as the inherent uncertainty of a variable's possible state. Mathematically entropy of a given variable $x$ is defined like below.

$$
h(x) = - \int p(x)log(p(x))
$$

Suppose a variable $x$ can only exist in 1 possible state. Then using the above equation the entropy is calculated like below.

$$
h(x) = -p(x)log(p(x)) \\
h(x) = -1 log 1 = 0
$$

Intuitively this makes sense. A variable that can only exist in one state then the uncertaintiy in the possible state of the variable must be zero. Things get more interesting when you add more states. For example, a variable than can exist in two states with equal probability of either state has the following entropy.

$$
h(x) = - [\frac{1}{2}\log(\frac{1}{2}) + \frac{1}{2}\log(\frac{1}{2})] \\
h(x) = - [\frac{1}{2}(-1) + \frac{1}{2}(-1)]\\
h(x) = - [-\frac{1}{2} - \frac{1}{2}]\\
h(x) = - [-1] = 1
$$

Hence, the entropy and inherent uncertaintity of this variable is 1. It is not immediately clear why this is true. The connection might be clearer if you're familiar with bits and binary numbers! A single bit of information can be used to represent two possible states. Thus, any variable that can exist in two possible states of equal probability requires at most a single bit of informationto fully define. Hence, the uncertainty of the variable is 1.

Needless to says, variables can be far more complex and in most cases, some states can be more probable than others. An unfair coin toss for example. In such a senario where heads has a probability of 25%, the entropy of the variable is defined like below.

$$
h(x) = - [Heads + Tails] \\
h(x) = - [\frac{1}{4}\log(\frac{1}{4}) + \frac{3}{4}\log(\frac{3}{4})] \\
h(x) = - [\frac{1}{4}(-2) + \frac{3}{4}(-0.415)]\\
h(x) = - [-0.5 - 0.311]\\
h(x) = - [-0.811] = 0.811
$$

The most interesting thing about this result is that it's less than when the possible states are uniformly distributed. Intuitively this also makes sense. A non-uniform distribution implies some states are more likely than others and thereofre, the uncertainty (entropy) of the variable must be smaller! Let's now take a look at how entropy is used in machine learning.

### Usage in Machine Learning

Entropy in machine learning is encountered where Cross Entropy is often used as a loss functions for classification tasks. For classification tasks, the goal is to train a model to map an input x to some probability for n-classes. In essence, the model is emulating a probability distribution, p(x), over the variables x. So how do we learn this distribution? If we assume that all of the n-classes are possible states that the variable x can exists in, it logically follows from the discussion above that the best distribution p(x) must be the distirbution with minimum entropy/uncertainity. In order to find the distribution p(x) we use the cross entropy loss function. The mathematical definition of cross entropy is given below.

$$
H(p, x) = - \sum_{x \in \mathcal{x}} p(x) log q(x)
$$


The cross entropy is comparing two different distributions p(x) and q(x) and computing how many more bits it takes to represent a state under q(x) instead of p(x). Another way to look at the cross entropy is that it is the expected value, under the true distribution, of the number of bits to encode some variable x. Because classification is often a supervised training problem, we can compute the true distribution p(x) while using the model to represent q(x). Thus by minimizing the cross entropy loss, we are slowly shifting q(x) to represent the true distribution p(x).


## Interesting observations

One interesting observation to see is that the definition of cross entropy constraints models to the training dataset. The cross entropy loss can only every truly model a distribution p(x) where the uncertainity in x is tied directly to the training dataset! Although it is not a shocking revelation for people used to training ML models, it's interesting to see the limitations enforced by the mathematics. One can partially get around this by having a truely representative dataset of the real world.


## ELBO

Later on 

On the connection between machine learning and compression <--- D: (Machine Learning is compression)

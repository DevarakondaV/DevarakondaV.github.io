---
layout: post
title: "A Brief Look at Entropy"
date: 2024-03-02
categories: AI
---

# Introduction

Lets take a brief look at entropy and its relevance in machine learning models. Entropy frequently appears in machine learning because it provides a foundation for training models to approximate probability distributions. For example, cross entropy loss is one of the most common loss function used for training classification models. 

## Entropy

Entropy, in the context of information theory, was introduced by Claude Shannon. Shannon defined entropy as the inherent uncertainty of a variable's possible state. Mathematically, the entropy of a given variable $$x$$ is defined below.

$$
h(x) = \sum_{x \in \mathcal{X}} p(x)\log{p(x)}
$$

Suppose a variable $$x$$ can only exist in one state. Then using the above equation, the entropy is calculated like below.

$$
\begin{align*}
& h(x) = -p(x)\log{p(x)} \\
& h(x) = -1 \log{1} = 0
\end{align*}
$$

Intuitively this makes sense. A variable that can only exist in one state has no uncertainty so the entropy/uncertainty in it's possible state must be zero. Things get interesting when you add more states. For example, a variable than can exist in two states with equal probability of either state has the following entropy.

$$
\begin{align*}
& h(x) = - \sum_{x \in \mathcal{X}}p(x) \log{p(x)} \\
& h(x) = - [\frac{1}{2}\log{\frac{1}{2}} + \frac{1}{2}\log{\frac{1}{2}}] \\
& h(x) = - [-\frac{1}{2} - \frac{1}{2}]\\
& h(x) = - [-1] = 1
\end{align*}
$$

Unlike before, it is not immediately clear why this is true. The connection is more clear when thinking in terms of bits and binary numbers. A single bit of information can be used to represent two possible states. Thus, any variable that can exist in two possible states of equal probability requires at most a single bit of information to fully describe. Hence, the uncertainty of the variable is one.

Unsurprisingly, variables can be far more complex and in most cases, some states can be more probable than others. For example, an unfair coin toss can be represented as a variable with two states with different probabilities. Suppose for instance that the probability of heads is 25%. Then the entropy of the variable is defined like below.

$$
\begin{align*}
& h(x) = - \sum_{x \in [Heads,\; Tails]}p(x) \log{p(x)} \\
& h(x) = - [Heads + Tails] \\
& h(x) = - [\frac{1}{4}\log(\frac{1}{4}) + \frac{3}{4}\log(\frac{3}{4})] \\
& h(x) = - [\frac{1}{4}(-2) + \frac{3}{4}(-0.415)]\\
& h(x) = - [-0.5 - 0.311]\\
& h(x) = - [-0.811] = 0.811
\end{align*}
$$

An interesting thing about this result is that it's less than if the possible states are uniformly distributed. A non-uniform distribution implies some states are more likely than others and therefore the inherent uncertainty of the variable must be smaller! Let's now take a look at how entropy is used in machine learning.

## Cross Entropy & Machine Learning

Although entropy is encountered frequently in ML, the most common occurrence is the cross entropy loss function. For classification tasks, the goal is to train a model to map an input $$x$$ to some distribution over n-classes. Under the assumption that all of the n-classes are possible states that the variable $$x$$ can exists in, it logically follows from the previous discussion that a good approximating distribution is one with minimum uncertainty. The cross entropy loss function can be used to train an ml-model to approximate this minimum uncertainty distribution. The mathematical definition of cross entropy is given below.

$$
H(p, x) = - \sum_{x \in \mathcal{X}} p(x) \log{q(x)}
$$

This definition looks very similar to the definition of entropy with a slight distinction in the probability distributions. The cross entropy is a metric that compares two distributions $$q(x)$$ and $$p(x)$$, by measuring how much more inefficient it is to represent states from $$p(x)$$ using $$q(x)$$. Because classification is often a supervised training problem, we can compute the true distribution $$p(x)$$ while using the ml-model to represent $$q(x)$$. Thus, minimizing the cross entropy loss will slowly shift $$q(x)$$ to represent the true distribution $$p(x)$$.

For example, suppose for a coin toss the $$p(x)$$ was 50/50. If we were training a machine learning model $$q(x)$$ and its current prediction were 25% heads and 75% tails, the cross entropy would be:

$$
\begin{align*}
& H(p,q)= - \sum_{x \in [Heads, Tails]} p(x) \log{q(x)}\\
& H(p,q) = - [Heads + Tails] \\
& H(p,q) = - [\frac{1}{2}\log{\frac{1}{4}} + \frac{1}{2}\log{\frac{3}{4}}]\\
& H(p,q) = - [-1.208]\\
& H(p,q) = 1.20
\end{align*}
$$

As expected, the coss entropy(1.20) is greater than the true entropy(1) for a fair coin toss! Now lets see what happens to the cross entropy if we had computed the gradients and updated the weights for the model and the new model predicts the probability of heads is 40%.

$$
\begin{align*}
& H(p,q)= - \sum_{x \in [Heads, Tails]} p(x) \log{q(x)}\\
& H(p,q) = - [Heads + Tails] \\
& H(p,q) = - [\frac{1}{2}\log{\frac{2}{5}} + \frac{1}{2}\log{\frac{3}{5}}]\\
& H(p,q) = - [-1.029]\\
& H(p,q) = 1.029
\end{align*}
$$

As we expect, the entropy is getting closer to 1 as the distribution $$q(x)$$ better approximates $$p(x)$$! This shows how the cross entropy loss can be used as a loss function for approximating probability distributions!
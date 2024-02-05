---
layout: post
title: "Exploring Self-Supervised Learning With DINO"
date: 2024-01-18
categories: AI
---

# Introduction


Knowledge distilliation with no labels (DINO) is a method of self-supervised learning.


# Knowledge Distilliation


What exactly is knowlege distilliation? Knowledge distilliation uses two different networks, a student and teacher, both of which output a probability distribution. The student is trained to match the output of the teacher. The training can be done under cross entropy. 


- Student Network: $g_{\theta_s}$
- Teacher Network: $g_{\theta_t}$
- Parameters: $\theta_s$, $\theta_t$
- Output Dimensions: K
- Output Distributions over K: $P_t$, $P_s$

Probabiltiy is obtained by temperature softmax like below.

$$
P_s(x)^{(i)} = \frac{exp(g_{\theta_s}(x)^{(i)}/\tau_s)}{\sum_{k=1}^K exp(g_{\theta_s}(x)^{(i)}/\tau_s)}
$$

The teacher network is fixed during training, while the cross-entropy is minimized with respect to the student netowrk.

$$
min_{\theta_s} H(P_{t}(x), P_s(x))
$$

# Self Supervised Learning

We adapt the above paradigm to self supervised learning.

1. For a given image x, generate a set V of different views using. These generated views are constructed using a multicrop strategy.
2. This set contain 2 global views of $x_1^g,x_2^g$ and several local views of smaller resolution.
3. All crops are passed through the student while global views are passed through the teacher. This encourages "local-to-global" corresondence. The following loss is minimize.

$$
min_{\theta_s}\sum_{x\in{x_1^g,x_2^g}}\sum_{x'\in V, x' \neq x}H(P_t(x), P_s(x'))
$$

DINO Standard

- 2 Global views of resolution: 224x224. Large > 50% area of the original image.
- Several local views of resolution: 96x96. Small < 50% area of the original image.


The Teacher network is built from past iterations of the student.

Freezing the teacher network over an epoch works pretty good while copying the student weight for the teacher fails to converge.


### Network Architecture

The backbone is either ViT or ResNet. In this article, we use ResNet and a projection head $h: g = h\;o\;f$

Model collapse is avoided by centering and sharpening of the momentum teacher outputs.


An extra learnable token $[CLS]$ is added to the sequence of embedding outputs of the initial linear layer. The Projection head $h$ is attached at it's output.


# Ref

Some of the articles on my page are based on summaries of research papers. All credit goes to the original authors of the paper!

1. 
---
layout: post
title: "VIT and MNIST"
date: 2023-05-23
categories: A.I
---

# ViT and MNIST

# Introduction

Lets see how ViT squares up against MNIST.


# ViT

ViT is a transformer model designed for images. In this post, I write some code to apply the model to MNIST to see how it performs compared to a standard CNN.

# Math

$$
\begin{align*}
& Image\;Height:\;H \\
& Image\;Width:\;W \\
& Image\;Channels:\;C \\
& Image\;Patch\;Size:\;P\\
& Image:\; x \in \mathcal{R}^{H \times  W \times C}\\
& Image\;Patch:\; x_p \in \mathcal{R}^{N \times (P^2C)}\\
& Number\;of\;Patches:\; N=\frac{HW}{P^2}\\
& Transformer\;Latent\;Vector\;Size:\; D\\
\end{align*}
$$

# Model

$$
\begin{align*}
& Number\;of\;Layers:\quad L\\
& LinearProjection:\quad  x_p^1\boldsymbol{E}\\
& Learnable\;Embedding:\quad x_{class}^D\\
& Learnable\;Embedding:\quad \boldsymbol{E}_{pos}\\
& \boldsymbol{E}\in\mathcal{R}^{(P^2C) \times D}\quad \boldsymbol{E}_{pos}\in\mathcal{R}^{(N + 1) \times D}\\
& z_0 = [x_{class};x_p^1\boldsymbol{E};x_p^2\boldsymbol{E};\dots;x_p^N\boldsymbol{E}] + \boldsymbol{E}_{pos}\\
& z_l^{'} = MultiHeadSelfAttention(LayerNorm(z_{l-1})) + z_{l-1}\qquad l=1...L\\
& z_l = MultiLayerPerceptron(LayerNorm(z_{l-1}^{'})) + z_l^{'} \qquad l=1...L\\
& y = LayerNorm(z_{L}^0)
\end{align*}
$$


# Data Preperation

#### MNIST

$$
\begin{align*}
& H = 28 \quad W = 28 \quad C = 1 \quad P = 7 \quad N = 16 \quad x = \mathcal{R}^{28 \times 28} \quad x_p = \mathcal{R}^{16 \times 49}\\
\end{align*}
$$

#### ViT Design

| Elements | Layer | Describe |
| --- | --- | --- |
| $$L_1$$ | Linear($$P^2$$, D) | $$x_i^{'} = L_1(x_p)$$ |
| $$L_2$$ | Linear(D, D) | $$x_{class}^{'} = L_2(x_{class})$$ |
| $$L_3$$ | Linear(N + 1, D) | $$x_{pos}^{'} = L_3(Norm([0,1,...16])) $$|
| $$z_0$$ | Embedding | $$z_0=[x_{class}^{'}, x_0^{'}, x_1^{'}, ..., x_{16}^{'}] + x_{pos}^{'}$$|
| $$LN_1^1$$ | LayerNorm | $$z_0^{LN11}=LN_1^1(z_0)$$ |
| $$MSA_1$$ | Multihead self attention | $$z_1^{'}=MSA_1(z_0^{LN11})+z_0^{LN11}$$|
| $$LN_1^2$$ | LayerNorm | $$z_1^{LN12}=LN_1^2(z_1^{'})$$ |
| $$MLP_1$$ | Multi layer perceptron | $$z_1=MSA(z_1^{LN12})+z_1^{LN12}$$|
| $$LN_2^1$$ | LayerNorm | $$z_1^{LN21}=LN_2^1({z_1})$$ |
| $$MSA_2$$ | Multihead self attention | $$z_2^{'}=MSA(z_1^{LN21})+z_1^{LN21}$$|
| $$LN_2^2$$ | LayerNorm | $$z_2^{LN22}=LN_2^2(z_2^{'})$$ |
| $$MLP_2$$ | Multi layer perceptron | $$z_2=MSA(z_2^{LN22})+z_2^{LN22}$$|
| $$LN_3$$ | LayerNorm | $$y=LN_3(z_2^0)$$ |
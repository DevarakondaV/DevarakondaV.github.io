---
layout: post
title: "Diffusion Models on MNIST"
date: 2024-01-15
categories: AI
---



### Introduciton

Let's take a look at diffusion models and train one on MNIST for fun! This post will be math heavy so buckle up!


### Lets define some variables and distributions distributions

- Data/Image: $\bold{x}_0$
- Latents of same dimension as $\bold{x}_0$: $\{\bold{x}_   1,...,\bold{x}_T\}$
- Reverse Process:$p_{\theta}(\bold{x}_{0:T}) := p(\bold{x}_T)\prod_{t=1}^{T}p_{\theta}(\bold{x}_{t-1}|\bold{x}_t)$
- Forward/Diffusion Process: $q(\bold{x}_{1:T}|\bold{x}) := \prod_{t=1}^{T}q(\bold{x}_{t}|\bold{x}_{t-1})$
- $p_{\theta}(\bold{x}_{t-1}|\bold{x}_t) := \mathcal{N}(\bold{x}_{t-1}; \bold{\mu}_{\theta}(\bold{x}_t,t),\sum_{\theta}(\bold{x}_t,t))$
- $q(\bold{x}_{t}|\bold{x}_{t-1}) := \mathcal{N}(\bold{x}_{t}; \sqrt{1-\beta_t}\bold{x}_{t-1},\beta_t\bold{I})$

### ELBO Diffusion

Lets take a look at where the loss function for training this model comes from. The ELBO

- Interested in: $p(X_{0:T})$
- Easier to Estimate: $p(X_{0:T},X_{1:T}) = p(X_{1:T}|X_{0:T})p(X_{0:T})$


We are going to estimate the above distribution us a family of distributions over the latents given by $q(X_{1:T}). Lets start the computation using the KL-divergence.

$$
D_{KL}(Q || P) = \int_{Q} q(X_{1:T}) \log{\frac{q(X_{1:T})}{p(X_{1:T}|X_{0:T})}}
$$

Well, we can rewrite this expression using properties of logaritim and bayes rule.

$$
\begin{align*}
D_{KL}(Q || P) &  = \int_{Q} q(X_{1:T}) [\log{q(X_{1:T})} - \log{p(X_{1:T}|X_{0:T})}]\\
& = \int_{Q} q(X_{1:T}) [\log{q(X_{1:T})} - \log{\frac{p(X_{0:T},X_{1:T})}{p(X_{0:T})}}]\\
& = \int_{Q} q(X_{1:T}) [\log{q(X_{1:T})} - [\log{{p(X_{0:T},X_{1:T})}} - \log{{p(X_{0:T})}}]]\\
& = \int_{Q} q(X_{1:T}) [\log{q(X_{1:T})} - \log{{p(X_{0:T},X_{1:T})}} + \log{{p(X_{0:T})}}]\\
& = \int_{Q} q(X_{1:T})\log{q(X_{1:T})} - \int_{Q} q(X_{1:T})\log{{p(X_{0:T},X_{1:T})}} + \int_{Q} q(X_{1:T})\log{{p(X_{0:T})}}\\
& = \int_{Q} q(X_{1:T})\log{\frac{q(X_{1:T})}{{p(X_{0:T},X_{1:T})}}} + \int_{Q} q(X_{1:T})\log{{p(X_{0:T})}}\\
& = \mathbb{E}_Q[\log{\frac{q(X_{1:T})}{{p(X_{0:T},X_{1:T})}}}] + \mathbb{E}_Q[\log{{p(X_{0:T})}}]
\end{align*}
$$

In the right term, the expectaion has no effect given the two distributions are independent. Lets get rid of it and rewrite

$$
= \mathbb{E}_Q[\log{\frac{q(X_{1:T})}{{p(X_{0:T},X_{1:T})}}}] + \log{{p(X_{0:T})}}\\
D_{KL}(Q || P) = -\mathbb{E}_Q[\log{\frac{{p(X_{0:T},X_{1:T})}}{q(X_{1:T})}}] + \log{{p(X_{0:T})}}
$$

Lets say we have a perfect estimator. This is okay because our goal is in effect to estimate the distribution perfectly. This means the KL-divegence is 0. Then we can write the above like below.

$$
- \log{{p(X_{0:T})}} = -\mathbb{E}_Q[\log{\frac{{p(X_{0:T},X_{1:T})}}{q(X_{1:T})}}]
$$

The right hand side of this expression is the ELBO. Thus, it must be greater than or equal to the the left hand side.

$$
- \log{{p(X_{0:T})}} \leq -\mathbb{E}_Q[\log{\frac{{p(X_{0:T},X_{1:T})}}{q(X_{1:T})}}]
$$

Take a look at the $p(X_{0:T},X_{1:T})$. This distribution is equivalent to $p(X_{0:T})$. Next, take a look at $q(X_{1:T})$. We've can also rewrite this distribution: $q(X_{1:T}|X_0)$

$$
- \log{{p(X_{0:T})}} \leq -\mathbb{E}_Q[\log{\frac{{p(X_{0:T})}}{q(X_{1:T}|X_0)}}]
$$

Now lets plug in our distributions from before.

$$
\begin{align*}
- \log{{p(X_{0:T})}} & \leq -\mathbb{E}_q[\log{\frac{{p(X_T)\prod_{t=1}^T p_{\theta}(x_{t-1}|x_{t})}}{\prod_{t=1}^T q(x_t|x_{t-1})}}] \\
& \leq - \mathbb{E}_q[\log{p(X_T)} + \log{\prod_{t=1}^T \frac{p_{\theta}(x_{t-1}|x_{t})}{q(x_t|x_{t-1})}}]\\
& \leq \mathbb{E}_q[-\log{p(X_T)} - \sum_{t\geq1} \log{\frac{p_{\theta}(x_{t-1}|x_{t})}{q(x_t|x_{t-1})}}] := L
\end{align*}
$$


## Training

During training the following values are set

During training, the goal is to learn $$p_{\theta}(x_{0:T})$$2

### Forward Process

- T = 1000
- Forward process variances are constants increasing linearly from $$\beta_1 = 10^{-4} \rightarrow \beta_T = 0.02$$
- Data scaled to [-1, 1]

### Reverse process

- UNet backbone
    - Weight normalization replaced with group normalization
    - 32x32 use 4 feature map resolutions
    - 2 convolutional residual blocks per resolution level, self attention at 16x16 resolutoin between the convolution blocks.
- Parameters are shared across time, which is specified to the network using the transformer sinusodial position embedding. Used as an embedding to represent the timestamp
- Self-Attention 16x16 feature map resolution.
- CIFAR 10 should have 35.7 million params
- CIFAR 10 dropout is 0.1
- CIFAR random horizontal flips
- learning rate 2E-4
- EMA on model params with decay factor 0.9999
- $$\sigma$$ is either $$\beta_t$$ or $$\frac{1-\bar{\alpha_t}-1}{1-\bar{\alpha_t}}\beta_t$$


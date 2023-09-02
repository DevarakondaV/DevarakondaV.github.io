---
layout: post
title: "Convolution: From Loops to Compiler"
date: 2020-06-02
categories: AI
---

# Introduction

Machine learning compliers have become a hot topic in the last few years. I thing the topic is fascinating and "pure" form of computer science. The effort to eek out every bit of performance out of a processor is computer science at it's finest. In that respect, in this article I explore how convolution is broken down from loops down to eventually the hardware on which it runs. Buckle up, because this is gonna be a long long story.


# Convolution Operation

When it comes to machine learning, we are interested in the convolution of two matrices. Additionally, traditional convolution requires flipping th kernel before performing the operation. In the context of neural networks, this flip is not necessary as the filters are learned anyway. Therefore, we can define the convolution of two matrices straight like below.

$$
\begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33} \\
\end{bmatrix}
\ast
\begin{bmatrix}
b_{11} & b_{12} & b_{13} \\
b_{21} & b_{22} & b_{23} \\
b_{31} & b_{32} & b_{33} \\
\end{bmatrix}
$$

The table below defines the output. Note that if the value lies outside the matrix, I use 0.

| Value | Equation |
| ----- | -------- |
| $c_{11}$ | $$b_{11}(0) + b_{12}(0) + b_{13}(0) + b_{21}(0) + b_{22}a_{11} + b_{23}a_{12} + b_{31}(0) + b_{32}a_{21} + b_{33}a_{22}$$ |
| $c_{12}$ | $$b_{11}(0) + b_{12}(0) + b_{13}(0) + b_{21}a_{11} + b_{22}a_{12} + b_{23}a_{13} + b_{31}a_{21} + b_{32}a_{22} + b_{33}a_{23}$$ |
| $c_{13}$ | $$b_{11}(0) + b_{12}(0) + b_{13}(0) + b_{21}a_{12} + b_{22}a_{13} + b_{23}(0) + b_{31}a_{22} + b_{32}a_{23} + b_{33}(0)$$ |
| $c_{21}$ | $$b_{11}(0) + b_{12}a_{11} + b_{13}a_{12} + b_{21}(0) + b_{22}a_{21} + b_{23}a_{22} + b_{31}(0) + b_{32}a_{31} + b_{33}a_{32}$$ |
| $c_{22}$ | $$b_{11}a_{11} + b_{12}a_{12} + b_{13}a_{13} + b_{21}a_{21} + b_{22}a_{22} + b_{23}a_{23} + b_{31}a_{31} + b_{32}a_{32} + b_{33}a_{33}$$ |
| $c_{23}$ | $$b_{11}a_{12} + b_{12}a_{13} + b_{13}(0) + b_{21}a_{22} + b_{22}a_{123} + b_{23}(0) + b_{31}a_{32} + b_{32}a_{33} + b_{33}(0)$$ |
| $c_{31}$ | $$b_{11}(0) + b_{12}a_{21} + b_{13}a_{22} + b_{21}(0) + b_{22}a_{31} + b_{23}a_{32} + b_{31}(0) + b_{32}(0) + b_{33}(0)$$ |
| $c_{32}$ | $$b_{11}a_{21} + b_{12}a_{22} + b_{13}a_{23} + b_{21}a_{31} + b_{22}a_{32} + b_{23}a_{33} + b_{31}(0) + b_{32}a_{21} + b_{33}a_{22}$$ |
| $c_{33}$ | $$b_{11}a_{22} + b_{12}a_{23} + b_{13}(0) + b_{21}a_{32} + b_{22}a_{33} + b_{23}(0) + b_{31}(0) + b_{32}a_{21} + b_{33}a_{22}$$ |

Notice that this definition differs from the true definition of convolution where the kernel is flipped. This is not relevant for neural networks because the kernal's are learned.

The convolution operation that is typically used in deep learning contains the following parameters.

| Parameter | Definition |
| --------- | ---------- |
| H | Image Height |
| W | Image Width |
| C | Image Channel |
| kSize | Kernel Size |
| nK    | Number of kernels |

We can perform this operation using a naive approach by simply running through the entire image and run using for loops. The code for this operation is given below.

```
// Convolution
for (int i = 0; i < nK; ++i){
    for(int j = 0; j < C; ++j){
        for(int h = 0; h < H; ++h) {
            for(int w = 0; w < W; ++w){
                for(int p = 0; p < kSize; ++p){
                    for(int q = 0; q < kSize; ++q){
                        Out[i][h][w] += kernels[i][j][p][q] * Image[j][h + p][w + q];
                    }
                }
            }
        }
    }
}
```

This code should make you scream. There is no real way to bypass this since we are forced by mathematical definition to run this operation as it is defined.

So how can we make this code run faster?? The trick is to maximize CPU utilizing while reducing memory access times. As usually, the operation that takes most time in CPUs is actually just moving data around. What we need to do is understand how to minimize data movement required to perform this operation.


I ran this code 20 times and average the time it takes: 18s. So now we have a baseline. Can we beat it? Lets try it out.

## CPU

The first step to optimizing this approach is to identity the CPU on which it is running.

| Parameter | Value |
| --------- | ----- |
| Model name | Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz |
| Thread(s) per core | 2 |
| Core(s) per socket | 6 |
| Socket(s) | 1 |
| Stepping | 10 |
| L1d cache | 192 KiB |
| L1i cache | 192 KiB |
| L2 cache |  1.5 MiB |
| L3 cache |  9 MiB |


We need to squeeze as much information into the L3 cache as possible while maintaining cache locality.

## Loop Tiling


## Sizing Tiles

## Automatic Tiling

### Polyhedra Model

### Integer Linear Programming

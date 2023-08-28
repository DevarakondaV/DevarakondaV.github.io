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
$$


We can perform this operation using a naive approch by simply running through the entire image and run using for loops. The code for this operation is given below.

```
// Convolution
for (int i = 0; i < kNum; ++i){
    for(int j = 0; j < kNum; ++j){
        for(int h = 0; h < kImSize; ++h) {
            for(int w = 0; w < kImSize; ++w){
                for(int p = 0; p < kKernel; ++p){
                    for(int q = 0; q < kKernel; ++q){
                        C[i][h][w] += weight[i][j][p][q] * input[j][h + p][w + q];
                    }
                }
            }
        }
    }
}
```

This code should make you scream. There is no real way to bypass this since we are mathematically forced to run this operations as it's defined. 

So how can we make this code run faster?? The trick is to maximize CPU utilizing while reducing memory access times?


## Loop Tiling


## Sizing Tiles

## Automatic Tiling

### Polyhedra Model

### Integer Linear Programming

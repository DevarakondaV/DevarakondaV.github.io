---
layout: post
title: "Diving into Automatic differentiation"
date: 2023-05-23
categories: AI
---
# Intro

In my graduate studies, I did a deep dive into autodifferentiation. I was struck by its simplicity and elegance, which inspired me to write this article. I was exploring how to write a library for training and inference on arithmetic circuits. Arithmetic circuits are a compiled Bayesian network that can use standard linear algebra operations. This allows us to train the circuit like a neural network using computational engines like TensorFlow or PyTorch. However, we wanted to decouple from these engines and use NumPy for differentiation and training. This gave me the opportunity to dive headfirst into autodifferentiation and truly appreciate the power of this technique.

# Theory

The foundational theory behind autodiff is the chain rule of differentiation. This rule is applied to composite functions. An example is given below.

$$
Composite\;Function:\;f\dot\;g = f(g(x))\\
Chain\;Rule:\;\frac{d}{dx}f(g(x)) = \frac{d}{dg}f(g(x))\frac{d}{dx}g(x)
$$

The chain rule is not limited by the depth of the composite. In fact, it can differentiate composite functions with any arbitrary depth. Below, I give an example of differentiating a composite functions of a composite function like $$f$$. Note that I redefined some functions using $$g_i$$.

$$
\begin{align*}
& f\dot\;g\dot\;h = f(g_2)\\
& g_0 = x \\
& g_1 = h(g_0) \\
& g_2 = g(g_1) \\
& \frac{d}{dx}f(g_2) = \frac{df}{dg_2}\frac{dg_2}{dx}\\
& \frac{d}{dx}f(g_2) = \frac{df}{dg_2}[\frac{dg_2}{dg_1}\frac{dg_1}{dx}]\\
& \frac{d}{dx}f(g_2) = \frac{df}{dg_2}[\frac{dg_2}{dg_1}[\frac{dg_1}{dg_0}\frac{dg_0}{dx}]]\\
\end{align*}
$$

Notice the recursive inherent in this definition. This recursion is exploited in order to compute the derivative of deep neural networks. Computing this is straightforward and a trivial example is given below.

$$
\begin{align*}
& g_0 = x\\
& h(x) = g_1 = g_0 \\
& g(g_1) = g_2 = 1 + 2g_1 \\
& f(g_2) = 2 + 4g_2\\
& \frac{df}{dg_2} = 4 \quad \frac{dg_2}{dg_1} = 2\\
& \frac{dg_1}{dg_0} = 1 \quad \frac{dg_0}{dx} = 1\\
\end{align*}\\
$$

$$
\frac{d}{dx}f(g(h(x))) = \frac{df}{dg_2}[\frac{dg_2}{dg_1}[\frac{dg_1}{dg_0}\frac{dg_0}{dx}]] = 4[2[1[1]]] = 8 
$$

#### Reverse Mode Auto differentiation

In the previous section, we discussed forward mode accumulation autodiff. Most deep learning libraries use an alternative strategy called reverse mode automatic differentiation (backpropagation). Although we will skip the details here, this strategy is significantly more efficient when dealing with a large number of trainable variables. In reverse mode, we differentiate the function with the innermost function first. An example is given below.


$$
\begin{align*}
& f\dot\;g\dot\;h = f(g_2)\\
& g_0 = x \\
& g_1 = h(g_0) \\
& g_2 = g(g_1) \\
& \frac{d}{dx}f(g(h(x))) = [\frac{df}{dh}]\frac{dh}{dx}\\
& \frac{d}{dx}f(g(h(x))) = [\frac{df}{dg}\frac{dg}{dh}]\frac{dh}{dx}\\
& \frac{d}{dx}f(g(h(x))) = [[\frac{df}{df}\frac{df}{dg}]\frac{dg}{dh}]\frac{dh}{dx}\\
\end{align*}
$$

Notice how the order of the operations in this approach is different from forward mode. The recursive relation is still present but the derivative with respect to the weights is now outside the parenthesis. This order is what improves the computational efficiency of the entire operation. Lets increase the complexity of the examples and introduce more variables. We introduce weights, represented by $$w_i$$, as the target to differentiate against.

$$
\begin{align*}
& y(w_1, w_2, x) = w_1 + w_2x\\
& g_1 = w_1 \\
& g_2 = w_2x \\
& g_3(g_1, g_2) = g_1 + g_2\\
& y(g_3) = g_3(g_1, g_2) \\
\end{align*}
$$

Before we dive into the math,  let's consider a problem that arises when we encounter $$g_3$$. It is a composite function of two variables $$g_1$$ and $$g_2$$. Although not obvious in this example, it is reasonable to expect that both $$g_1$$ and $$g_2$$ could be dependent on the same weight. Additionally, even if they did not depend on the same weight, it is not possible to know at run time which functions and weights are coupled in very large neural networks. In this scenario, how could one find the derivative of $$y$$ with respect to $$w_1$$? The answer is to apply the total derivatives rule.

$$
\frac{\partial f(w,g_1,g_2,...g_n)}{\partial w} = \frac{\partial f}{\partial w}\frac{\partial w}{\partial w} + 
\frac{\partial f}{\partial g_1}\frac{\partial g_1}{\partial w} +
\frac{\partial f}{\partial g_2}\frac{\partial g_2}{\partial w} +
...
\frac{\partial f}{\partial g_n}\frac{\partial g_n}{\partial w}
=
\frac{\partial f}{\partial w} + \sum_{i = 1}^{n} \frac{\partial f}{\partial g_i}\frac{\partial g_i}{\partial w}
$$

Now we can differentiate by applying both the chain rule and the total derivative rule.

$$
\frac{\partial y}{\partial w_1} = 
[\frac{\partial y}{\partial g_1}]\frac{\partial g_1}{\partial w_1} + 
[\frac{\partial y}{\partial g_2}]\frac{\partial g_2}{\partial w_1}\\
\frac{\partial y}{\partial w_2} = 
[\frac{\partial y}{\partial g_1}]\frac{\partial g_1}{\partial w_2} + 
[\frac{\partial y}{\partial g_2}]\frac{\partial g_2}{\partial w_2}
\\
\frac{\partial y}{\partial w_1} = 
[[\frac{\partial y}{\partial g_3}]\frac{\partial g_3}{\partial g_1}]\frac{\partial g_1}{\partial w_1} + 
[[\frac{\partial y}{\partial g_3}]\frac{\partial g_3}{\partial g_2}]\frac{\partial g_2}{\partial w_1}\\
\frac{\partial y}{\partial w_2} = 
[[\frac{\partial y}{\partial g_3}]\frac{\partial g_3}{\partial g_1}]\frac{\partial g_1}{\partial w_2} + 
[[\frac{\partial y}{\partial g_3}]\frac{\partial g_3}{\partial g_2}]\frac{\partial g_2}{\partial w_2}
$$

$$
\frac{\partial g_1}{\partial w_1} = 1 \quad
\frac{\partial g_2}{\partial w_1} = 0 \quad
\frac{\partial g_3}{\partial g_1} = [1 + 0] \quad
\frac{\partial y}{\partial g_3} = 1\\
\frac{\partial g_1}{\partial w_2} = 0 \quad
\frac{\partial g_2}{\partial w_2} = x \quad
\frac{\partial g_3}{\partial g_2} = [0 + 1] \quad
\frac{\partial y}{\partial g_3} = 1\\
$$


$$
\begin{align*}
\frac{\partial y}{\partial w_1} = 
[[\frac{\partial y}{\partial g_3}]\frac{\partial g_3}{\partial g_1}]\frac{\partial g_1}{\partial w_1} + 
[[\frac{\partial y}{\partial g_3}]\frac{\partial g_3}{\partial g_2}]\frac{\partial g_2}{\partial w_1} = [[1][1 + 0][1] + [[1][0 + 1]][0] = 1\\
\frac{\partial y}{\partial w_2} = 
[[\frac{\partial y}{\partial g_3}]\frac{\partial g_3}{\partial g_1}]\frac{\partial g_1}{\partial w_2} + 
[[\frac{\partial y}{\partial g_3}]\frac{\partial g_3}{\partial g_2}]\frac{\partial g_2}{\partial w_2} = [[1][1 + 0]][0] + [[1][0 + 1]][x] = x\\
\end{align*}
$$

Does this output make sense? Yeah! You can confirm this by taking the derivative of $$y$$ directly.

$$
\nabla_{w _i} y = 
\begin{bmatrix}
\frac{\partial}{\partial{w_1}}(w_1 + w_2x) && \frac{\partial}{\partial{w_2}}(w_1 + w_2x) 
\end{bmatrix}
= 
\begin{bmatrix}1 && x\end{bmatrix}\\ 
$$

#### Vectors and Matricies

We now take a look at applying the above theory to vectors and matricies. All machine learning models, from transformers to decision trees, can be decomposed into basic linear algebra operations. Therefore, if we can apply the theory discussed above to matrices and operations on matrices, we can effectively differentiate any arbitrary-sized neural network. This is where the true power of autodiff becomes shines through.


Lets start off by looking at a more advanced example using vector functions. For vector functions, we can apply the vector chain rule of differentiation.

$$
\begin{align*}
& \vec{y} = 
\begin{bmatrix}
y_1 \\
y_2
\end{bmatrix}
=
\begin{bmatrix}
f_1 \\
f_2 \\
\end{bmatrix}
=
\begin{bmatrix}
w_1 + w_2x \\
sin(w_1x^3)
\end{bmatrix}
\end{align*}
$$

Notice that the above equation requires us to use both the chain rule and the total derivative rule. Additionally, we introduce $$sin$$ function as an example of an activation function.

$$
\begin{align*}
& g_1 = w_1 \\
& g_2 = w_2x \\
& g_3 = w_3x^3 \\
& g_4 = g_1 + g_2 \\
& g_5 = sin(g_3)\\
& \vec{y}(g_1, g_2, g_3) =
\begin{bmatrix}
g_4(g_1,g_2) \\
g_5(g_3)
\end{bmatrix}\\
\end{align*}
$$

Applying the chian rule:

$$
\begin{align*}
& \frac{\partial \vec{y}}{\partial w_1} =
\begin{bmatrix}
\frac{\partial y_1}{\partial g_1}\frac{\partial g_1}{\partial w_1} + 
\frac{\partial y_1}{\partial g_2}\frac{\partial g_2}{\partial w_1} +
\frac{\partial y_1}{\partial g_3}\frac{\partial g_3}{\partial w_1} 
\\
\frac{\partial y_2}{\partial g_1}\frac{\partial g_1}{\partial w_1} + 
\frac{\partial y_2}{\partial g_2}\frac{\partial g_2}{\partial w_1} +
\frac{\partial y_2}{\partial g_3}\frac{\partial g_3}{\partial w_1}
\end{bmatrix}
\end{align*}
$$

Notice that the above matrix can be decomposed.

$$
\begin{align*}
& \frac{\partial \vec{y}}{\partial w_1} =
\begin{bmatrix}
\frac{\partial y_1}{\partial g_1} &
\frac{\partial y_1}{\partial g_2} &
\frac{\partial y_1}{\partial g_3}
\\
\frac{\partial y_2}{\partial g_1} &
\frac{\partial y_2}{\partial g_2} &
\frac{\partial y_2}{\partial g_3}
\end{bmatrix}
\begin{bmatrix}
\frac{\partial g_1}{\partial w_1} \\
\frac{\partial g_2}{\partial w_1} \\
\frac{\partial g_3}{\partial w_1}
\end{bmatrix}
=
\frac{\partial \boldsymbol{y}}{\partial \boldsymbol{g}}
\frac{\partial \boldsymbol{g}}{\partial w_1}
\end{align*}
$$

There are two interesting things to observe here. First, we have calculated the gradient of $$\vec{y}$$ as a matrix multiplication between two Jacobian matrices.  Second, notice that the two matrices are effectively decoupled. Each component ($$\frac{\partial y_i}{\partial g_i}$$) of the first matrix can itself be represented as a matrix multiplication! This sets up the recursive relationship between Jacobian matrices like the one shown below!

$$
\begin{align*}
& \frac{\partial \vec{y}}{\partial w_1} =
\frac{\partial \boldsymbol{y}}{\partial \boldsymbol{g}}
\frac{\partial \boldsymbol{g}}{\partial w_1}
\end{align*} = 
[\frac{\partial \boldsymbol{y}}{\partial \boldsymbol{h}}\frac{\partial \boldsymbol{h}}{\partial \boldsymbol{g}}]
\frac{\partial \boldsymbol{g}}{\partial w_1}
$$

Finally, notice that the derivative with respect to the weight $$w_1$$ is always outside $$\frac{\partial \boldsymbol{y}}{\partial \boldsymbol{g}}$$. This means that $$\frac{\partial \boldsymbol{y}}{\partial \boldsymbol{g}}$$ is same for all weights, which saves an enormous amount of computational resources when computing the gradients for all weights! But, we can do even better by computing the gradients with respect to all weights simultaneous. We compute the derivative of the function with respect to the weight vector $$\vec{w}$$. This only leads to a minor change where $$\frac{\partial \boldsymbol{g}}{\partial w_1}$$ and $$\frac{\partial \vec{y}}{\partial \vec{w}}$$ are now matrices. Most generally the gradient of $$\vec{y}$$ with respect to $$\vec{w}$$ is given by below.

$$
\begin{align*}
& \frac{\partial \vec{y}}{\partial \vec{w}} =
\begin{bmatrix}
\frac{\partial y_1}{\partial g_1} &
\dots &
\frac{\partial y_1}{\partial g_{m}}
\\
\vdots & \ddots & \vdots \\
\frac{\partial y_k}{\partial g_1} &
\dots &
\frac{\partial y_k}{\partial g_{m}}
\end{bmatrix}
\begin{bmatrix}
\frac{\partial g_1}{\partial w_1} & \dots & \frac{\partial g_1}{\partial w_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial g_m}{\partial w_1} & \dots & \frac{\partial g_m}{\partial w_n}
\end{bmatrix}
\end{align*}
$$

Now lets continue with the the example and make sure we get the expected gradient. For brevity, I ignored the recursive relations but the approach should be clear.

$$
\begin{align*}
& \frac{\partial \vec{y}}{\partial \vec{w}} =
\begin{bmatrix}
\frac{\partial y_1}{\partial g_1} &
\frac{\partial y_1}{\partial g_2} &
\frac{\partial y_1}{\partial g_3}
\\
\frac{\partial y_2}{\partial g_1} &
\frac{\partial y_2}{\partial g_2} &
\frac{\partial y_2}{\partial g_3}
\end{bmatrix}
\begin{bmatrix}
\frac{\partial g_1}{\partial w_1} & \frac{\partial g_1}{\partial w_2} &\frac{\partial g_1}{\partial w_3} \\
\frac{\partial g_2}{\partial w_1} & \frac{\partial g_2}{\partial w_2} &\frac{\partial g_2}{\partial w_3} \\
\frac{\partial g_3}{\partial w_1} & \frac{\partial g_3}{\partial w_2} &\frac{\partial g_3}{\partial w_3}
\end{bmatrix}
\\
& \frac{\partial \vec{y}}{\partial w_1} =
\begin{bmatrix}
[\frac{\partial y_1}{\partial g_4}][\frac{\partial g_4}{\partial g_1}] &
[\frac{\partial y_1}{\partial g_4}][\frac{\partial g_4}{\partial g_2}] &
[\frac{\partial y_1}{\partial g_4}][\frac{\partial g_4}{\partial g_3}]
\\
[\frac{\partial y_2}{\partial g_5}][\frac{\partial g_5}{\partial g_1}] &
[\frac{\partial y_2}{\partial g_5}][\frac{\partial g_5}{\partial g_2}] &
[\frac{\partial y_2}{\partial g_5}][\frac{\partial g_5}{\partial g_3}]
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0\\
0 & x & 0\\
0 & 0 & x^3
\end{bmatrix}
\\
& \frac{\partial \vec{y}}{\partial w_1} =
\begin{bmatrix}
[1][1 + 0] &
[1][0 + 1] &
[1][0]
\\
[1][0] &
[1][0] &
[1][cos(g_3)]
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0\\
0 & x & 0\\
0 & 0 & x^3
\end{bmatrix}
\\
& \frac{\partial \vec{y}}{\partial w_1} =
\begin{bmatrix}
1 &
1 &
0
\\
0 &
0 &
cos(g_3)
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0\\
0 & x & 0\\
0 & 0 & x^3
\end{bmatrix}
=
\begin{bmatrix}
1 & x & 0 \\
0 & 0 & x^3cos(w_3x^3)
\end{bmatrix}
\end{align*}
$$

Is this correct? Lets compute the jacobian without the chain rule.

$$
\begin{align*}
& \boldsymbol{J}(\vec{y}) =
\begin{bmatrix}
\frac{\partial y_1}{\partial w_1} & \frac{\partial y_1}{\partial w_2} & 
\frac{\partial y_1}{\partial w_3} \\
\frac{\partial y_2}{\partial w_1} & \frac{\partial y_2}{\partial w_2} & \frac{\partial y_2}{\partial w_3} \\
\end{bmatrix}
=
\begin{bmatrix}
1 & x & 0 \\
0 & 0 & x^3cos(w_1x^3) \\
\end{bmatrix}
\end{align*}
$$

Finally, notice that for one of the partial derivatives we need to evaluate $$cos(g_3)$$. Values like this, which appear in the jacobian, are intermediate values in the calculation of the gradient. One can calculate these values while evaluating the function $$\vec{y}$$, a forward pass! Then in the backwards pass, these values can be used to compute the gradient! 

# Operations

Lets take a look at some basic operations and their gradients in code. In the equations below, $$J$$ represents the jacobian from a previous operation. If there are no previous operations then this value is simply a matrix of ones.

#### Addition

Lets start off by looking at adding two matricies and their derivative.

$$
C = A + B \\
\frac{\partial C}{\partial A} = I + 0 = \boldsymbol{J}I \\
\frac{\partial C}{\partial B} = 0 + I = \boldsymbol{J}I
$$


```
class Add:
    def init(self):
        pass
    
    def forward(self, A, B):
        C = A + B
        self.jacobianA = I # <----- Intermediate Value
        self.jacobinaB = I # <----- Intermediate Value
        return C
    
    def backward(self, jacobian):
        return jacobian * self.jacobianA, jacobian * jacobianB

A = np.zeros(5,5)
B = np.zeros(5,5)

Op = Add()
Op.forward(A,B)
JacobianA, JacobianB = Op.backward(1)
```

#### Subtraction

$$
C = A - B \\
\frac{\partial C}{\partial A} = I + 0 = \boldsymbol{J}I \\
\frac{\partial C}{\partial B} = 0 - I = -\boldsymbol{J}I
$$

```
class Sub:
    def init(self):
        pass
    
    def forward(self, A, B):
        C = A - B
        self.jacobianA = I
        self.jacobianB = -I
        return C
    
    def backward(self, prev):
        return jacobian * self.jacobianA, jacobian * self.jacobianB

A = np.zeros(5,5)
B = np.zeros(5,5)

Op = Sub()
Op.forward(A,B)
JacobianA, JacobianB = Op.backward(1)
```

#### Multiplication

$$
C = A * B \\
\frac{\partial C}{\partial A} = \boldsymbol{J}B \\
\frac{\partial C}{\partial B} = \boldsymbol{J}A
$$

```
class Multiply: # Hadamard Product
    def init(self):
        pass
    
    def forward(self, A, B):
        C = np.multiply(A,B)
        self.jacobianA = B
        self.jacobianB = A
        return C
    
    def backward(self, jacobian):
        return jacobian * self.jacobianA, jacobian * jacobianB
A = np.zeros(5,5)
B = np.zeros(5,5)

Op = Multiply
Op.forward(A,B)
JacobianA, JacobianB = Op.backward(1)
```

#### Matrix Multiplication

Differentiating a matrix multiplication is a little tricky. I've given the solution below for brevity.

$$
C = AB \\
\frac{\partial C}{\partial A} = \boldsymbol{J}B^T \\
\frac{\partial C}{\partial B} = A^T\boldsymbol{J}
$$

```
class MatMul:
    def init(self):
        pass
    
    def forward(self, A, B):
        C = A * B
        self.jacobianA = np.transpose(B)
        self.jacobianB = np.transpose(A)
        return C
    
    def backward(self, jacobian):
        return [jacobian * self.jacobianA, self.jacobianB * jacobian]

A = np.zeros(5,5)
B = np.zeros(5,5)
Op = MatMul()
Op.forward(A,B)
JacobianA, JacobianB = Op.backward(1)
```

#### Chaining Operations

Here, I give an example of chaining operations together and computing the gradient of the entire function. One can think of this as a "tiny" neural network containing some operations. Below, we compute $$f = (A + B)*C$$ in the forward pass. In the backward pass, we compute the gradient of $$f$$ with respect to $$A$$ and $$C$$. 

```
A = np.array()
B = np.array()
C = np.array()
addOp = Add()
multiplyOp = Multiply()

# Forwards
D = addOp.forward(A,B)
E = multiplyOp.forward(D, C)

# Backwards
[jacobianD, jacobianC] = multiplyOp.backward(np.ones(E.shape))
[jacobianA, jacobianB] = addOp.backward(jacobianD)

# jacobianA ---- Derivative of f with respect to A
# jacobianB ---- Derivative of f with respect to B
```

#### Conclusion

In this post, we explored the very basics of how autodifferentiation works in most machine learning libraries. We saw how autodiff can be used to compute the gradients of complex functions, such as deep neural networks, with relative ease. We also talked very briefly about computational cost of different approaches to autodiff. Incredibly, we can get even further computation efficiency by exploiting mathematical symmetries and sparsities present in Jacobians when applying autodiff. But that is a conversation for a different time; here we explore the fascinating simplicity and structure of autodifferentiation.

#### Ref

1. [The Matrix Calculus You Need For Deep Learning](https://arxiv.org/abs/1802.01528)
2. [Differentiate Matrix Multiplication](https://math.stackexchange.com/questions/1866757/not-understanding-derivative-of-a-matrix-matrix-product)




<!-- #### PREVIOUS Linear Example

Lets increase the complexity of our functions by introducing more variables. We introduce $$w_i$$ which refers to "weights" that one would see in neural networks. Now we differentiate with respect to the weights of the function as one would exepect in neural networks. What we're actually computing is the gradient of the function with respect to its weights.

$$
\begin{align*}
& y(w_1, w_2, x) = w_1 + w_2x\\
& \nabla_{w _i} y = 
\begin{bmatrix}
\frac{\partial}{\partial{w_1}}(w_1 + w_2x) && \frac{\partial}{\partial{w_2}}(w_1 + w_2x) 
\end{bmatrix}\\ 
& g_1 = w_1 \\
& g_2 = w_2x \\
& g_3 = g_1 + g_2\\
& y = g_3 \\
& \frac{\partial y}{\partial w_1} = \frac{\partial y}{\partial g_3}\frac{\partial g_3}{\partial w_1}\\
& \frac{\partial y}{\partial w_1} = \frac{\partial y}{\partial g_3}[\frac{\partial g_3}{\partial g_1}\frac{\partial g_1}{\partial w_1} + \frac{\partial g_3}{\partial g_2}\frac{\partial g_2}{\partial w_1}]\\
& \frac{\partial y}{\partial w_1} = \frac{\partial y}{\partial g_3}[\frac{\partial g_3}{\partial g_1}[\frac{\partial g_1}{\partial w_1}] + \frac{\partial g_3}{\partial g_2}[\frac{\partial g_2}{\partial w_1}]]\\
& \frac{\partial y}{\partial w_2} = \frac{\partial y}{\partial g_3}[\frac{\partial g_3}{\partial g_1}[\frac{\partial g_1}{\partial w_2}] + \frac{\partial g_3}{\partial g_2}[\frac{\partial g_2}{\partial w_2}]]\\
\end{align*}
$$

$$
\frac{\partial g_2}{\partial w_1} = 0 \quad \frac{\partial g_1}{\partial w_1} = 1 \quad \frac{\partial g_3}{\partial g_2} = 1 \quad \frac{\partial g_3}{\partial g_1} = 1 \\
\frac{\partial g_2}{\partial w_2} = x \quad \frac{\partial g_1}{\partial w_2} = 0 \quad \frac{\partial g_3}{\partial g_2} = 1 \quad \frac{\partial g_3}{\partial g_1} = 1 \\
\frac{\partial y}{\partial g_3} = 1 \\
\begin{align*}
& \frac{\partial y}{\partial w_1} = \frac{\partial y}{\partial g_3}[\frac{\partial g_3}{\partial g_1}[\frac{\partial g_1}{\partial w_1}] + \frac{\partial g_3}{\partial g_2}[\frac{\partial g_2}{\partial w_1}]]\\
& \frac{\partial y}{\partial w_1} = 1[1[1] + 1[0]] = 1\\
& \frac{\partial y}{\partial w_2} = 1[1[0] + 1[x]] = x\\
& \nabla_{w _i} y = 
\begin{bmatrix}
1 && x
\end{bmatrix}\\ 
\end{align*}
$$

Although, not apparent from this example, more complex functions will introduce some subtle issues that lead to greater computation complexitiy when calculating the gradient of the function. For example, suppose we define $$y$$ like below. Note that this function is non-linear due to $$x^2$$.

$$
\begin{align*}
& y(w_1, w_2, x) = w_1x * w_2x\\
& \nabla_{w _i} y = 
\begin{bmatrix}
\frac{\partial}{\partial{w_1}}(w_1xw_2x) && \frac{\partial}{\partial{w_2}}( w_1xw_2x) 
\end{bmatrix}\\
& g_1(w_1,x) = w_1x \\
& g_2(w_2,x) = w_2x \\
& g_3 = g_1g_2\\
& y = g_3 \\
& \frac{\partial y}{\partial w_1} = \frac{\partial y}{\partial g_3}\frac{\partial g_3}{\partial w_1}\\
& \frac{\partial y}{\partial w_1} = \frac{\partial y}{\partial g_3}[\frac{\partial g_3}{\partial g_1}\frac{\partial g_1}{\partial w_1}]\\
& \frac{\partial y}{\partial w_2} = \frac{\partial y}{\partial g_3}[\frac{\partial g_3}{\partial g_2}\frac{\partial g_2}{\partial w_2}]\\
\end{align*} \\
\frac{\partial g_2}{\partial w_1} = x \quad \frac{\partial g_1}{\partial w_1} = 1 \quad \frac{\partial g_3}{\partial g_2} = g_1 \quad \frac{\partial g_3}{\partial g_1} = g_2 \\
\frac{\partial g_2}{\partial w_2} = x \quad \frac{\partial g_1}{\partial w_2} = 1 \quad \frac{\partial g_3}{\partial g_2} = g_1 \quad \frac{\partial g_3}{\partial g_1} = g_2 \\
\frac{\partial y}{\partial g_3} = 1 \\
\frac{\partial y}{\partial w_1} = 1[g_2 * x] = g(w_2,x) * x\\
\frac{\partial y}{\partial w_2} = 1[g_1 * x] = g(w_1,x) * x\\
$$

Suppose we want to train the function above. A naive implementation of auto-diff may be done like below.

1. compute $$y(w_1,w_2,x)$$
    1. This step is the forward pass of the neural network.
2. compute $$\nabla_{w _i} y$$
    1. Requires us to compute $$g_1(w_1,x)$$ and $$g_2(w_2,x)$$.

Astute readers would notice that computing $$y(w_1, w_2, x)$$ already requires us to compute $$g_1(w_1,x)$$ and $$g_2(w_2,x)$$. Additionally, notice how we can only compute the gradient with respect to each weight, one at at a time. This requires us to traverse the operations multiple times, one time for each variable. This is due to the fact that order of the operaitons in our gradient requires us to compute from the innermost to the outter most function. This is incredibly inefficient for a large number of variables. -->
---
layout: post
title: "Reinforcement Learning With Snake!"
date: 2020-06-02
categories: AI
---

### SSPlayer

{:refdef: style="text-align: center;"}
![yay](/assets/gifs/snake9.2.gif)
![yay](/assets/gifs/snake9.gif)
{: refdef}

SSPlayer is an implementation of Deep Q-Network, as described in "Playing Atari with Deep Reinforcement Learning" by DeepMind, to play games. [Github](https://github.com/DevarakondaV/SSPlayer), I also wrote [Medium](https://medium.com/@devarakonda.vishnu5/simple-deep-q-learning-with-math-1afb0cfdcf0d) post about the mathematics invovled in deep Q-learning. I used SSPlayer to train neural networks to play snake!

# Results

After running the training operation for about a 200 thousand states where the greed gradually reduces from 100% to 10% over the first 100 thousand states, the result of how well the network learned to play snake can be seen above.

Clearly, this is not the optimal solution for playing the game. Still, we can see the network has learned some policy that is effective at guiding the snake to its food. A longer training period would improve the performance of the network. : )

---
layout: post
title: "Detecting a Smile in Video"
date: 2020-06-02
categories: AI
---

<div style="text-align: center; margin-bottom: 25px;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/wJKl1L7pMMA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</div>

# Detecting Smiles : )

I wrote some code to detect smiles in a video. This was probably my first real ML project after doing the introductory MNIST project. [Github TF](https://github.com/DevarakondaV/tflow_smile), [Github OpenCV](https://github.com/DevarakondaV/tflow_smile)

### Model
The model is quite simple. It uses a haar cascades face detection algorithm to detect a face. After some post processing, this image is passed to a simple CNN that performs logistic regression to determine if the subject is smiling. Easy! I used OpenCV for the face detector and Tensorflow/OpenCV for the neural network. The results can be seen above.

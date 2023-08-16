---
layout: post
title: "Modeling a Graph Database on Dynamodb."
date: 2023-08-15
categories: Software
---


# Introduction

I've recently been more fascinated by graphs! They're a simple but powerful data structure for modeling data and relationships. Almost all social media companies use graphs to model relationships like friends, follows, etc. I wanted to explore graph data modeling by building a graph database on top of dynamodb. 

#### Preliminary

Graphs are made up of vertices and edges. In a graph database, on can define objects as vertices and the relationships between objects using edges. For instance, one can define a user objects and use "Friend" relations as edges between the user objects. What makes graph modeling powerful is how quickly you can answer questions like: "Who are UserA's friends?". 
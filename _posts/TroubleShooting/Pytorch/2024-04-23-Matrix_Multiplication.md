---
title: "[TroubleShooting-Pytorch] WX + b vs. XW + b"
excerpt: "왜 머신러닝/딥러닝 교재들과 딥러닝 프레임워크의 표현식이 다를까?"

categories: "troubleshooting-pytorch"
tags:
    - pytorch
    - Matrix
toc: true  
toc_sticky: true
toc_label: "Contents In Page"
author_profile: true
use_math: true
date: 2024-04-23
---

## WX + b vs. XW + b
- 블로그에 그간 배운 것들을 정리하다가 문득 궁금증이 생겼다. "왜 교재에서는 $\mathbf{W}\mathbf{X} + \mathbf{b}$ 로 표현하는데, Pytorch 나 Tensorflow 같은 딥러닝 프레임워크에서는 $\mathbf{X}\mathbf{W} + \mathbf{b}$ 로 표현할까?"
![Untitled](/assets/images/TroubleShooting/matrixmul.png){: .align-center}


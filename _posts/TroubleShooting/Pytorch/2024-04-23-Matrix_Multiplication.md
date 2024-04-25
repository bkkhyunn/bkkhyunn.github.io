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

- 우리가 머신러닝이나 딥러닝을 할 때 가지고 있는 것은 parameter 인 $W$ 보다는 데이터인 $X$ 를 가지고 있다.
- 그리고 $X$ 는 행의 개수가 데이터의 개수(혹은 batch_size), 열의 개수가 feature 의 개수를 나타낸다. 데이터 분석에서의 관습이다.
- [데이터 사이언스 스쿨](https://datascienceschool.net/02%20mathematics/02.02%20%EB%B2%A1%ED%84%B0%EC%99%80%20%ED%96%89%EB%A0%AC%EC%9D%98%20%EC%97%B0%EC%82%B0.html#id16)에서는 다음과 같이 이야기한다.
> **하나의 데이터 레코드를 단독으로 벡터로 나타낼 때는 하나의 열(column)**로 나타내고 **복수의 데이터 레코드 집합을 행렬로 나타낼 때는 하나의 데이터 레코드가 하나의 행(row)**으로 표기하는 것은 얼핏 보기에는 일관성이 없어 보이지만 추후 다른 연산을 할 때 이런 모양이 필요하기 때문이다. 데이터 분석에서 쓰는 일반적인 관례이므로 외워두어야 한다.
- 그렇다면 $W$ 또한 행이 output 의 개수, 열이 feature 의 개수를 나타내야 한다. 왜냐하면 가중치는 우리가 풀고 싶은 함수에 근사하도록 구해야 할 값으로 feature 의 개수와 같아야 하기 때문이다.
  - 단적인 예로 키와 몸무게로 수명을 예측하고 싶을 때 feature 는 키와 몸무게가 되고, 각 키와 몸무게가 가진 가중치에 따라 예측값이 달라진다. 이 가중치를 적절히 구하면 수명을 잘 예측할 수 있을 것이다.
  - 또한 행이 output 의 개수가 되면 메모리적으로도 효율성을 챙길 수 있다. 이 [포스트](https://bkkhyunn.github.io/troubleshooting-pytorch/pytorch_for_HPC/)에 그 이유가 정리되어 있다.
- 딥러닝 연산의 근간이 되는 행렬곱셈은 벡터의 내적으로 이루어져 있고, 앞 행렬의 열의 수와 뒤 행렬의 행의 수가 일치해야 한다.
- 그러면 $WX$ 와 $XW$ 의 차이는 같은 값이 출력된다고 했을 때 다음과 같다.
  - transpose 되는 행렬이 달라진다. $WX$ 는 $X$ 가 transpose 되고, $XW$ 는 $W$ 가 transpose 된다.
  - 출력의 shape 이 반대가 된다.
  ![Untitled](/assets/images/TroubleShooting/matmul.jpeg){: .align-center}*그림 1*
- Linear Regression 에 대한 Neural Network Diagram 을 그렸을 때 더 눈에 들어오는 차이를 느낄 수 있다.
  ![Untitled](/assets/images/TroubleShooting/matmul2.jpeg){: .align-center}*그림 2*

- 위 그림을 보면서 다음과 같은 결론을 내릴 수 있겠다.
  1. 우리는 $X$의 shape을 관습으로 정의해놨고 Pytorch 와 같은 딥러닝 프레임워크에서는 $n$ 을 batch size 로 활용하고 있다.
  2. 하나의 데이터 레코드는 열벡터이다. 따라서 그림 2의 $W$ 는 (3) 혹은 (3,1) 로 표기할 수 있다.
  3. 우리는 $y$ 값도 하나의 column 으로 둔다. 종속'변수'이기 때문이다.
  4. 그렇다면 그림 1에서 실제값과 예측값의 비교 측면으로 볼 때 $XW$ 가 $WX$ 보다 유리하다. $y$ 값은 행의 개수로 $n$ 을 가지고 있을 것이기 때문이다.
  5. 그림 2를 보면, $y$ 가 1개이기 때문에 $W$는 열벡터이고, 행렬과 벡터의 곱 연산을 생각해볼 때 $XW$ 가 $WX$ 보다 효율적이다. 왜냐하면 $X$ 의 shape 은 우리가 정하고 있고, 그 $X$ 와 행렬곱을 위해 차원을 맞춰주는 transpose 연산이 없기 때문에이다.

- 중요한 건 우리가 1) **행렬 $X$ 에 대한 모양을 이미 관습으로 정의하고 있다는 것(batch_size, the number of features)**과 2) 차원을 쉽게 맞출 수 있어 **행렬 연산의 편의성**을 가져올 수 있기 때문에 PyTorch 와 같은 딥러닝 프레임워크에서는 $XW$ 의 순서를 따르는 것이다.
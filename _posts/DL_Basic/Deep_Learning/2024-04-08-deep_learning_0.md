---
title: "[Deep Learning] 0. 딥러닝 기본"
excerpt: "딥러닝을 공부할 때 어떻게 할지와, Neural Network"

categories: "deep_learning"
tags:
    - deep learning
toc: true  
toc_sticky: true
toc_label: "Contents In Page"
author_profile: true
use_math: true

date: 2024-04-08
---

## 딥러닝은
- 머릿속으로 생각하는 아이디어를 구현하고 결과를 뽑는 **Implementation Skills(TF, Pytorch)**도 중요하지만, 동시에 **Math Skills(Linear Algebra, Probability)** 도 당연히 중요하다.
        
    ![Untitled](/assets/images/DL_basic/Untitled.png){: .align-center}
        
    - **데이터**, **모델**, **loss function**, **알고리즘(optimizer)** → 이 4가지 항목에 비추어서 논문을 보자!
      - loss function 은 모델을 어떻게 학습시킬지에 대한 것(파라미터 업데이트를 위한 기준값)
        
        → loss 는 이루고자 하는 것에 proxy(근사치)이다. 이 loss 를 왜 사용해야 하는지, 이 loss 가 줄어드는 것이 내가 풀고자 하는 문제를 어떻게 푸는지 이해해야 한다.
        
        → loss function 의 목적은 단순히 줄이는 것이 아니라, **내 모델이 학습하지 않은 데이터에 잘 동작**하도록 하는 것이다!
        
        ![Untitled](/assets/images/DL_basic/Untitled 1.png){: .align-center}
        
      - Optimization Algorithm 은 다양한 테크닉이 있다.(Dropout, Early Stopping, k-fold validation, Weight decay, Batch Normalization, Mixup, Ensemble, Bayesian Optimization)
       
        → 네트워크가 단순히 학습 데이터가 아닌 실환경에서 잘 동작하도록 도와준다!

    - 임팩트가 있던 딥러닝 방법론들!
        
        - 2012 - AlexNet → 딥러닝 패러다임의 시작! 딥러닝이 실제로 성능을 발휘함
        - 2013 - DQN → 강화학습 방법론. Q-Learning. 딥마인드를 있게함
        - 2014 - Encoder/Decoder(seq2seq), Adam(optimizer) → Adam 은 웬만하면 잘된다!
        - 2015 - GAN, ResNet → 딥러닝이 왜 딥러닝인지, 네트워크를 깊게 쌓을 수 있게 만들어준 ResNet
        - 2017 - Transformer(Attention Is All You Need)    
          → Transformer 구조는 정말 중요! MHA(Multi Head Attention) 등의 구조를 이해. 다른 기존의 RNN 에 비해 어떤 장점, 좋은 성능이 있는지 이해!
        - 2018 - Bert → fine-tuned NLP models.
        - 2019 - Big Language Models(GPT-X)
        - 2020 - Self-Supervised Learning (SimCLR)      
          → visual representation 이미지를 컴퓨터가 잘 이해할 수 있는 벡터로 바꾸는 것.(내 학습 데이터 외에 추가 데이터(라벨이 없는)까지 활용해서!)
    
## Neural Networks & Multi-Layer Perceptron
- Neural Networks 란? 애매하게 사람의 뇌를 모방한 컴퓨팅 시스템.
  - Neural Net 은 선형 & 비선형 연산이 반복적으로 일어나는 function approximators 다! 즉, 함수를 근사하는 것이다.
  - 내가 풀고자하는 어떤 문제(함수)를 모방하는 함수가 뉴럴 네트워크다.
  - 나의 파라미터가 어느 방향으로 움직였을 때 loss func 이 줄어드는 지를 찾고 그 방향으로 파라미터를 바꾸는 것이 목적이다.
        
    ![Untitled](/assets/images/DL_basic/Untitled%202.png)
        
    ![Untitled](/assets/images/DL_basic/Untitled%203.png)
        
    ![Untitled](/assets/images/DL_basic//Untitled%204.png)
        
    - 어려운 점은 단순히 linear 한 변환만 있는 것이 아니라는 점이다.
    - 행렬에 대한 해석: 두 개의 벡터 스페이스 사이의 변환. 선형성을 가지는 변환이 있을 때, 그 변환은 항상 행렬로 표현된다. 
    - 어떤 행렬을 찾겠다는 것은 두 개의 벡터 스페이스(두 차원) 사이에 선형 변환을 찾겠다는 것.
        
        ![Untitled](/assets/images/DL_basic/Untitled%205.png)
        
  - 네트워크가 표현하는 표현력을 극대화하기 위해서
    - 단순히 선형 결합을 n 번 반복하는 게 아니라 한번 선형변환(affine)이 오면 activation func 을 이용해서 Non-linear transform 을 거치고, 그렇게 얻어지는 feature vec 을 다시 선형변환, 비선형변환 거치는 것을 n 번 반복하면 더 많은 표현력을 가진다. 이게 보통의 뉴럴 네트워크!
    - 어떤 활성화 함수(비선형 변환)가 있어야 딥러닝에서 의미가 있지만, 어떤 함수가 좋은지는 모른다. 상황마다 다르기 때문에 공부와 경험이 필요하다.

  - 앞으로 공부할 때
    - 어떤 loss func 이 어떤 성질을 가지고, 이게 왜 내가 원하는 결과를 얻어낼 수 있는지를 반드시 알아야 한다!
    - 모든 Loss 가 내가 원하는 문제를 완벽히 푼다고 할 수 없기 때문이다. 예를 들어 회귀에서 보통 사용되는 MSE 는 이상치가 있을 때 그 이상치를 맞추려고 하다 보면 네트워크가 망가질 수도 있다!

    - 분류에서 보통 사용되는 CE(Cross Entropy) 도 왜 분류에서 잘 되는지 알아야 한다.
      - 일반적으로 분류의 아웃풋은 one-hot vector 이다. 클래스가 10개면 10차원의 벡터가 나오는 것. 이 때 정답만 1이고 나머지는 0.
      - CE Loss 를 최소화 하는 것은, $\hat{y}$ 을 보통 logit 이라고 하는데, 나의 뉴럴 네트워크의 출력값 중에서 정답 클래스에 해당하는 값만 높힌다는 것이다. 즉 그 정답 차원(정답 dim)에 해당하는 출력값을 키우는 게 우리의 목적이다.
      - 우리가 분류를 한다고 했을 때, d 개의 라벨을 가진다고 한다면 d개의 아웃풋 중에서 제일 큰 값을 가지는 출력의 index 만을 고려한다. 그 값이 얼마나 큰지는 중요하지 않다. 분류를 잘하는 관점에서는 다른 값들 대비 정답의 값이 높기만 하면 된다. 그것을 수학적으로 표현하기 까다롭기 때문에 분류 문제를 풀 때 CE 를 사용하는 것이다.
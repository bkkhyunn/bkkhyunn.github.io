---
title: "[Deep Learning] 0. Neural Network"
excerpt: "Neural Network(NN)에 대해 알아보자"

categories: "deep_learning"
tags:
    - deep learning
    - Perceptron
    - Activation Function
    - Neural Network
toc: true  
toc_sticky: true
toc_label: "Contents In Page"
author_profile: true
use_math: true

date: 2024-04-08
---

## 딥러닝의 역사
![Untitled](/assets/images/DL_basic/history.png)*딥러닝의 연혁*{: .asign-center}
- 자세한 역사를 알기보다 딥러닝이 어떻게 발전해왔고 어떻게 지금 내가 공부하는 방향으로 흘러오게 되었는지를 이해하면 도움이 된다.

## Neural Network
- Neural Networks 란? 애매하게 사람의 뇌를 모방한 컴퓨팅 시스템. 딥러닝의 근간이 된다.
- 계산 신경과학 분야에서는 뇌를 정확히 모델링하는 것에 도전하면서 인공 신경망의 핵심 유닛을 단순화된 생물학적 뉴런으로 비유하는 것을 지속해왔다.
- 기존의 선형 분류나 회귀(Regression)모델에서 1)**활성화 함수(activation func)의 추가**와 2)**층(layer)으로 순서대로 뉴런을 연결하는 구조**를 통해서 완전한 신경망을 만들 수 있다.
- 초기 인공신경망인 퍼셉트론부터 시작해서, MLP(multi layer perceptron), 오늘날에는 인공 신경망을 복잡하게 쌓아올린 딥러닝이 뜨겁게 발전하고 있다!
 
### Perceptron
- 1950년대 후반에 프랭크 로젠블라트가 제안한 초기의 인공 신경망으로, **다수의 신호를 입력으로 받아 하나의 결과를 내보내는 알고리즘**이다.
- 개별 신경세포는 간단한 연산능력이 있다고 알려져 있는데, 이 신경세포 하나하나가 서로 연결되어 동작하면 고도의 인지, 판단 능력이 생성된다. 이러한 동작과 비슷하게 수학적으로 모델링한 것이 퍼셉트론이라 볼 수 있다.
![Untitiled](/assets/images/DL_basic/perceptron.png)*뉴런과 퍼셉트론*{: .asign-center}
- 퍼셉트론 신호는 1 또는 0의 두가지 값을 가질 수 있다. 즉, 어떠한 **임계값**을 기준(hard thresholding func, step func)으로 하여 **신호를 활성화하거나, 비활성화**할 수 있다.
- 원래는 각 입력값과 그에 해당하는 가중치의 곱의 전체 합이 임계치(threshold)를 넘으면, 종착지에 있는 인공 뉴런은 출력 신호로서 1을 출력하고 그렇지 않을 경우에는 0을 출력하는데, 이 임계치를 입력단으로 넘겨 입력값을 1로, 가중치 처럼 곱해지는 변수로 편향(bias)을 표현한다.
![Untitiled](/assets/images/DL_basic/perceptron2.png)*$x$는 입력값, $w$는 가중치(weight), $y$는 출력값. bias($w_0$) 또한 입력값이다.*{: .asign-center}
- 아래 그림과 같이 퍼셉트론의 식을 나타낼 수 있고, 이 퍼셉트론을 활용하여 and, or 연산이 가능하다. 즉 퍼셉트론이 $w$ 와 $bias$($b$) 에 따라서 and 와 or 을 흉내낼 수 있다는 것이다.
![Untitiled](/assets/images/DL_basic/perceptron3.png)*퍼셉트론 그림(좌), 퍼셉트론 식(우)*{: .asign-center}
![Untitiled](/assets/images/DL_basic/perceptron4.png)*하나의 퍼셉트론으로 and 와 or 연산을 수행할 수 있다.*{: .asign-center}
- threshold func(step func) 처럼 인공 뉴런의 출력값을 변경시키는 함수를 **활성화 함수(activation func, $\phi$)** 라고 부른다.
- 실제 신경 세포 뉴런에서의 신호를 전달하는 축삭돌기의 역할을 퍼셉트론에서는 **가중치(weights)**가 대신한다. 각 입력값에 해당하는 가중치의 값이 크면 클수록 해당 입력 값이 중요하다는 것을 의미한다.
- **편향(bias)**은 가중치와 상관없이 뉴런이 1 또는 0이 되는 성향을 제어한다. 높은 편향은 뉴런이 출력 1을 만들기 위해 더 많은 입력을 필요로 한다는 의미이고 낮은 편향은 더 쉽게 출력 1을 만든다는 뜻이다.

### 활성화 함수(Activation Function)
- 초기 인공 신경망 모델인 퍼셉트론은 활성화 함수로 계단 함수(step func)를 사용하였지만, 그 뒤에 등장한 여러가지 발전된 신경망들은 계단 함수 외에도 여러 다양한 활성화 함수를 사용하기 시작했다.
- 시그모이드 함수나 소프트맥스 함수 또한 활성화 함수 중 하나. 퍼셉트론의 활성화 함수는 계단 함수이지만 여기서 활성화 함수를 시그모이드 함수로 변경하면, 퍼셉트론은 곧 이진 분류를 수행하는 로지스틱 회귀와 동일하다. 다시 말하면 로지스틱 회귀 모델이 인공 신경망에서는 하나의 인공 뉴런이 된다. **로지스틱 회귀를 수행하는 인공 뉴런과 위의 퍼셉트론의 차이는 오직 활성화 함수의 차이이다.**
- 따로 포스트를 두어 Relu 등의 다양한 Activation Function 에 대해 공부할 것이다. 여기서는 아래와 같은 활성화 함수들이 있다는 것만 알고 넘어가자.
- **중요한 것은 선형으로 표현되는 퍼셉트론에 비선형을 추가하여 표현력을 늘려주는 것이 활성화 함수!**
![Untitiled](/assets/images/DL_basic/activate.png)*활성화함수에는 다양한 종류가 있다.*{: .asign-center}

### 단층 퍼셉트론(single layer perceptron)
- 지금까지 살펴본 Perceptron 은 단층 퍼셉트론이다. 단층 퍼셉트론은 입력층(input layer)과 출력층(output layer) 두 단계로만 이루어져 있다.
![Untitled](https://wikidocs.net/images/page/24958/%ED%8D%BC%EC%85%89%ED%8A%B8%EB%A1%A03.PNG){: .align-center}
- 이 단층 퍼셉트론으로 위에서 봤듯 AND, OR, NAND 문제를 풀 수 있지만, 선 하나로 XOR 문제는 풀 수 없다. 즉 **단층 퍼셉트론은 linear 문제만 풀 수 있고, XOR 과 같은 non-linear 문제는 풀 수 없는 것**이다.
- 실생활을 근사할 수 있는 함수들은 대부분이 non-linear 이다. 따라서 단층 퍼셉트론만으로는 절대적으로 한계가 있다.
![Untitled](/assets/images/DL_basic/non-linear.png)*XOR Problem*{: .align-center}
- 이 문제를 풀기 위해서는 적어도 두 개의 선이 필요한데, 이에 대한 해답이 **다층 퍼셉트론(MLP)**이다.

### 다층 퍼셉트론(MLP, multi layer perceptron)
- 다층 퍼셉트론과 단층 퍼셉트론의 차이는 단층 퍼셉트론은 입력층과 출력층만 존재하지만, 다층 퍼셉트론은 중간에 층을 더 추가했다.
- 입력층과 출력층 사이에 존재하는 층을 **은닉층(hidden layer)**이라 한다. 이 은닉층이 어떻게 구성되었는가에 따라 딥러닝 모델의 성능을 좌우할 수 있다.
![Untitled](/assets/images/DL_basic/mlp.png){: .align-center}
- 단순히 퍼셉트론 두 개를 stacking 하면 XOR 문제가 풀린다.
- MLP 부터는 ANN(Artificial Neural Network)라고도 불리고, hidden layer 가 2개 이상인 신경망을 DNN(Deep Neural Network)라고 부른다.

### 신경망 학습방법
- 위처럼 퍼셉트론이 OR, AND, XOR 문제를 풀 때 가중치를 바꿔가면서 풀었는데, 기계가 가중치를 스스로 찾아내도록 자동화시키기 위해서 **training** 혹은 **learning** 단계가 필요하다.
- 이 때 다음 포스트에서 다룰 선형, 로지스틱 회귀에서 사용되는 **손실 함수(loss func)**와 **옵티마이저(optimizer)**가 사용된다.
- 학습시키는 인공 신경망이 DNN 일 경우 **딥러닝(Deep Learning)**이라고 하는 것이다.
- 다음 포스트에서 주요한 **역전파(Backpropagation)**에 대해 자세히 알아보기로 하고, 딥러닝의 흐름을 이해하고 어떻게 학습이 되는지 이해하기 위해서 과거의 신경망 학습방법이 어떻게 발전되어 왔는지 간단히 보자.

  #### Hebb's rule (Hebbian Learning)
  - 신경망에서 뉴런 사이의 weight 들을 업데이트 시켜주는 메커니즘 중 하나이다. Hebb 이 신경망을 학습시키기 위해 제안했다.
  -  Hebbian learning을 요약하면, "cell A가 cell B를 자극시키는 데 충분하고, 지속적인 원인으로 작용한다면, cell B에 대한 cell A의 작용력이 내부적으로 변화를 일으켜 증가하게 된다." 라고 말할 수 있다. 즉 만일 **어떤 신경세포의 활성이 다른 신경세포가 활성하는 데 계속적으로 공헌한다면**, 두 신경세포 간의 **연결 가중치를 증가**시켜야 한다는 것이다.
  -  $W_{ij}^{new} = W_{ij}^{old}+{\alpha}a_ib_j$
  -  $\alpha$ 는 학습률이고, $a_i$ 와 $b_j$ 가 A 와 B 뉴런의 활성화 여부(1 또는 0)이다. **두 뉴런이 같이 활성화가 되면 두 뉴런 사이의 가중치가 계속해서 업데이트** 된다.
  -  즉 Hebb's rule 에서 **학습은 뉴런들 간의 연결을 만드는 활동**이고 하나의 퍼셉트론 입장에서 **input** 과 **output** 을 가지고 가중치를 업데이트한다. 이는 **Unsupervised Learning** 이다.
  -  Hebbian Learning 은 이처럼 Unsupervised Learning 이지만(뉴런 활성화 여부가 가중치를 업데이트 하기 때문), 이후 목적이 되는 **정답값**을 이용하면서 Supervised Learning 으로 발전하고, 오늘날 딥러닝 학습의 근간이 되었다.

  #### Perceptron learning rule
  - 만일 어떤 신경세포의 활성으로 다른 신경세포가 **잘못된** 출력을 낸다면, 두 신경세포 간의 연결 가중치를 오차에 비례하여 조절히여 올바른 예측을 한다는 아이디어.
  ![Untitled](/assets/images/DL_basic/perceptron_learning_rule.png){: .align-center}
  - 두 개의 뉴런 간의 weight를 학습할 때 **목적패턴**(정답값)과 예측값을 이용하는 학습 방법. 목적패턴과 예측값이 다를 때 입력이 1인 뉴런에 대해서만 오차만큼 학습을 시킨다. 이 때 같은 데이터를 여러번 반복해서 해를 찾는 과정을 거친다.
  ![Untitled](/assets/images/DL_basic/perceptron_learning_rule2.png)*$t$ 는 실제값, $z$ 는 예측값. 활성화 함수는 step func 이다.*{: .align-center} 
  - Perceptron learning rule은 기존의 Hebbian Learning 에서 목적 패턴(정답값)이라는 개념을 도입하여 Supervised Learning 을 한 것으로 보인다. 그렇게 **잘못된 출력에 대해서만 가중치를 조정**한다.
  - **그러나 이 Perceptron rule 은 단층 퍼셉트론에서 사용되었기 때문에 XOR같은 선형 분리가 불가능한 문제에 대해서는 적절히 학습할 수가 없었다.**
  - 즉 모든 training example들이 linearly separable해야 한다. 만약 linearly separable하지 않다면, 잘 수렴할지 확신할 수 없다.

  #### Delta rule (Widrow Hoff Learning Algorithm)
  - 다층 퍼셉트론과 같은 신경망의 학습에 사용된다.
    ![Untitled](/assets/images/DL_basic/delta_rule.png){: .align-center} 
  - delta rule 의 핵심은 **gradient descent**를 이용하는 것이다. gradient descent는 이 후 포스트에서 다룰 **역전파(backpropagation)**의 근간이기 때문에 중요하다.
  - Adaline 은 Perceptron 을 발전시킨 인공신경망이다.
    - Perceptron의 경우에는 순입력 함수의 출력값을 임계값을 기준으로 1과 -1(혹은 1, 0)로 분류한다. 가중치의 업데이트는 입력에 대한 활성함수의 출력값(1 또는 -1)과 실제 결과값이 같은지 다른지에 따라 이루어진다.
    ![Untitled](/assets/images/DL_basic/perceptron-adaline1.png){: .align-center}
    - **Adaline**은 **순입력함수의 리턴값과 실제 결과값의 오차(Error)가 최소화** 되도록 가중치를 조정하게 된다.
    - 또한 Adaline 은 활성화함수를 거치면 Continous value 를 가진다. 활성화함수를 거친 출력(예측값)을 실제값과 비교할 때 loss func 으로 MSE 를 활용했다.
    ![Untitled](/assets/images/DL_basic/perceptron-adaline2.png){: .align-center} 
  - **Gradient Descent**는 파라미터 $W$에 대한 error를 계산하는 목적함수(Objective Func, Loss Func)을 잘 정의하고, 이 함수가 최소화되는 방향을 찾아서 점점 나아가는 것이다.
  - 구체적으로는 error func 을 각 파라미터로 **편미분**하여, 각 파라미터에 대한 기울기를 구해 **local minima** 를 향하도록 움직여간다.
  - Delta Rule은 Perceptron Rule과 거의 유사하나, 전체 학습 데이터에 대한 **오차를 최소화**하는 방법이다.
- 이렇게 발전하여 오늘날 gradient descent, back propagation 이 활용되고 있다.

## 다시 Neural Network
- Neural Net 은 **선형** & **비선형** 연산이 반복적으로 일어나는 function approximators 다! 즉, 내가 풀고자 하는 어떤 문제이자 함수를 근사하는 함수이다.
- **나의 파라미터가 어느 방향으로 움직였을 때 loss func 이 줄어드는 지**를 찾고 **그 방향으로 파라미터를 바꾸는 것**이 목적이다.
      
  ![Untitled](/assets/images/DL_basic/Untitled%202.png){: .align-center}
      
  ![Untitled](/assets/images/DL_basic/Untitled%203.png){: .align-center}
      
  ![Untitled](/assets/images/DL_basic//Untitled%204.png){: .align-center}
      
- 딥러닝이 어려운 점은 단순히 linear 한 변환만 있는 것이 아니라는 점이다.
  - 딥러닝에서는 loss func 이 local minima 로 향하도록 하는 파라미터($W$, weights)들을 찾는 것이 목적이고, 이 파라미터들은 행렬로 표현된다.
  - 행렬에 대한 해석: 두 개의 벡터 스페이스 사이의 변환. 선형성을 가지는 변환이 있을 때, 그 변환은 항상 행렬로 표현된다. 어떤 행렬을 찾겠다는 것은 두 개의 벡터 스페이스(두 차원) 사이에 **선형 변환**을 찾겠다는 것이다.
  ![Untitled](/assets/images/DL_basic/Untitled%205.png){: .align-center}
  - 이 때 위에서 보았든 현실세계의 문제를 더 잘 근사하기 위해서는 non-linearity 가 필요하고 이를 위해 MLP 에서는 퍼셉트론을 2개 이상 stack 했다.
  ![Untitled](/assets/images/DL_basic/nonlinear.png)*$\rho$ 는 acitvation func 이다.*{: .align-center} 
  - 이처럼 네트워크가 표현하는 표현력을 극대화하기 위해서 단순히 선형 결합을 n 번 반복하는 게 아니라 한번 선형변환(affine)이 오면 **activation func 을 이용해서 Non-linear transform** 을 거치고, 그렇게 얻어지는 feature vec 을 다시 선형변환, 비선형변환 거치는 것을 **n 번 반복**하면 **더 많은 표현력**을 가진다. 이것이 보통의 Neural Network 이다!
  - 어떤 활성화 함수(비선형 변환)가 있어야 딥러닝에서 의미가 있지만, 어떤 함수가 좋은지는 모른다. 상황마다 다르기 때문에 공부와 경험이 필요하다.

- 앞으로 공부할 때
  - 어떤 **loss func** 이 어떤 성질을 가지고, 이게 왜 내가 원하는 결과를 얻어낼 수 있는지를 반드시 알아야 한다! 모든 loss 가 내가 원하는 문제를 완벽히 푼다고 할 수 없기 때문이다.
  - 예를 들어 회귀에서 보통 사용되는 MSE 는 이상치가 있을 때 그 이상치를 맞추려고 하다 보면 네트워크가 망가질 수도 있다!

  💡 분류에서 보통 사용되는 CE(Cross Entropy) 도 왜 분류에서 잘 되는지 알아야 한다. 일반적으로 분류 task 의 아웃풋은 **one-hot vector**이다. 클래스가 10개면 10차원의 벡터가 나오는 것이고, 이 때 정답만 1이고 나머지는 0이다. CE Loss 를 최소화 하는 것은, $\hat{y}$ 을 보통 logit 이라고 하는데, 나의 Neural Network의 출력값 중에서 정답 클래스에 해당하는 값만 높힌다는 것이다. 즉, 그 정답 차원(정답 dim)에 해당하는 출력값을 키우는 게 우리의 목적이다. 우리가 분류를 한다고 했을 때, $d$개의 라벨을 가진다고 한다면 $d$개의 아웃풋 중에서 **제일 큰 값을 가지는 출력**의 **index**만을 고려한다. **그 값이 얼마나 큰지는 중요하지 않다.** 분류를 잘하는 관점에서는 다른 값들 대비 정답의 값이 높기만 하면 된다. 그것을 수학적으로 표현하기 까다롭기 때문에 분류 문제를 풀 때 CE 를 사용하는 것이다.
  {: .notice--warning}

## 딥러닝은
- 머릿속으로 생각하는 아이디어를 구현하고 결과를 뽑는 **Implementation Skills(TF, Pytorch)**도 중요하지만, 동시에 **Math Skills(Linear Algebra, Probability)** 도 당연히 중요하다.
      
  ![Untitled](/assets/images/DL_basic/Untitled.png){: .align-center}
      
  - **데이터**, **모델**, **loss function**, **알고리즘(optimizer)** → 이 4가지 항목에 비추어서 논문을 보자!
    - loss function 은 모델을 어떻게 학습시킬지에 대한 것(파라미터 업데이트를 위한 기준값)
      
      → **loss 는 이루고자 하는 것에 proxy(근사치)**이다. 이 loss 를 왜 사용해야 하는지, 이 loss 가 줄어드는 것이 내가 풀고자 하는 문제를 어떻게 푸는지 이해해야 한다.
      
      → loss function 의 목적은 단순히 줄이는 것이 아니라, **내 모델이 학습하지 않은 데이터에 잘 동작**하도록 하는 것이다!
      
      ![Untitled](/assets/images/DL_basic/Untitled 1.png){: .align-center}
      
    - Optimization Algorithm 은 다양한 테크닉이 있다.(Dropout, Early Stopping, k-fold validation, Weight decay, Batch Normalization, Mixup, Ensemble, Bayesian Optimization)
     
      → 네트워크가 단순히 학습 데이터가 아닌 **실환경**에서 잘 동작하도록 도와준다!

  - 임팩트가 있던 딥러닝 방법론들! 다른 포스트에서 모두 다룰 것!
      
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
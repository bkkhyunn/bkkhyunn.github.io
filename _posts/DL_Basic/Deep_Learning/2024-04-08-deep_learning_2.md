---
title: "[Deep Learning] 2. Linear Neural Networks for Classification"
excerpt: "Softmax Regression 에 대해 알아보자"

categories: "deep_learning"
tags:
    - deep learning
    - Logistic Regression
    - Softmax Regression
    - Cross Entropy
toc: true  
toc_sticky: true
toc_label: "Contents In Page"
author_profile: true
use_math: true

date: 2024-04-08
---

앞에서 다룬 Linear Regression 과 Linear Classification 이 엄밀히 딥러닝의 하위분야는 아니지만, 전통적인 머신러닝 방식이고 이를 이해하면 딥러닝에 대한 한층 더 심도있는 이해를 할 수 있다. 따라서 여기서는 전 포스트에 이어 Linear Classification 을 정리해보자.<br>
Computer Vision 분야의 Image Classification 뿐만 아니라, Object Detection 이나 Segmentation 도 결국에는 **분류**다. 물론 Object Detection 에서는 물체의 위치를 regression 으로 예측하지만, 그 안에 있는 class 에 대해서는 classification 을 통해 결과가 나온다. 따라서 앞으로의 무궁무진한 Classification 을 이해하기 위해서 반드시 정복하고 가자.

## Linear Regression 으로 분류를 하지 않는 이유
- Linear Regression 을 사용하여 분류 task 를 수행할 수 있지만, 권장되지 않는다. 왜 그럴까?
- 바로 앞 포스트처럼 Squared Error 를 사용하는 Linear Regression 의 경우, 제곱 형식으로 큰 Loss 를 크게 반영하여 학습에 도움이 되지만 이상치에 영향을 많이 받는다고 공부했다.
- 새로운 데이터가 들어왔는데 그게 이상치에 해당하면, 찾아낸 최적의 선이 크게 영향을 받게 된다. 또한 분류 task 는 Regression 과 달리 이산적인 값(e.g. 남자/여자, 강아지/고양이 등)의 출력을 필요로 한다.
- 따라서 분류 task 에서 효과적인 문제 해결방법이 아니다.
- 이를 해결하기 위해 Classification 으로 이진 분류에 사용되는 **Logistic Regression** 이 등장했고, 다중 분류로 일반화된 **Softmax Regression** 이 등장했다.

## Classification(분류)
- 카테고리 별로 값을 할당하거나, 어떤 카테고리에 속할 확률이 얼마나 되는지를 예측하는 것이 **분류**이다.
- 회귀(Regression) vs. 분류(Classification)
  - 둘 다 학습에 정답값이 있는 **지도학습**에 속한다.
  - **회귀** 모델은 예측값으로 **연속적인 값**을 출력하고, **분류** 모델은 예측값으로 **이산적인 값**을 출력한다. 쉽게, 회귀는 값 예측이고 분류는 클래스 예측이다.
- 현실에서는 어떤 카테고리에 해당하는지를 예측하는 문제를 더 많이 접하게 된다. "그렇다", "그렇지 않다" 의 이진분류를 수행할 수도 있고, 여러가지 클래스를 분류하는 다중분류일 수도 있다.
- 분류에는 해당 데이터가 무엇으로 분류되어야 하는지 **정답 레이블(label)**이 필요하다. 컴퓨터는 인간의 언어를 이해하지 못하기 때문에 이 레이블(label)을 컴퓨터가 이해할 수 있는 숫자형으로 인코딩한다. 그 방식에는 두 가지가 있다.
  - **Label Encoding**: 문자열의 Unique 값을 숫자로 바꿔주는 방법이다. 강아지, 고양이, 닭을 각각 숫자 1,2,3 으로 바꿔주는 것이다. 즉, $y \in \lbrace 1, 2, 3 \rbrace$ 로 정의될 수 있다.
  - **One-hot Encoding**: long-type 으로 구성된 변수들을 wide-type 으로 바꾸어 변수의 차원을 늘려준다. 그러면 분류 밑에 있던 강아지, 고양이, 닭이 각각 변수가 되고 해당 동물에 해당하면 1, 아니면 0이 되는 이진수를 가진다. 즉 $y \in \lbrace (1,0,0), (0,1,0), (0,0,1) \rbrace$ 으로 정의될 수 있다.
  - label encoding 은 숫자의 크기(ordinal 특성)가 반영되고, 숫자값을 가중치로 잘못 인식하여 값에 왜곡이 생길 수 있다. 따라서 분류 task 에서는 one-hot vector 를 많이 사용한다.
- 분류 task 를 Linear Model 로 어떻게 나타낼 수 있을까?

  <center>$\begin{aligned}o_1 &= x_1 w_{11} + x_2 w_{12} + x_3 w_{13} + x_4 w_{14} + b_1,\\
    o_2 &= x_1 w_{21} + x_2 w_{22} + x_3 w_{23} + x_4 w_{24} + b_2,\\
    o_3 &= x_1 w_{31} + x_2 w_{32} + x_3 w_{33} + x_4 w_{34} + b_3\end{aligned}$</center>
  - 위처럼 분류 task에서는 클래스 개수(위 예제에서는 3개) 만큼의 선형변환식을 가진다. 4개의 feature($x_j$)와 3개의 클래스($o_i$)가 있고, 각 클래스의 출력값에 대응하는 선형변환 함수의 가중치($w_{ij}$)와 편향($b_i$)으로 구성된다.
  - 다이어그램으로 표현하면 아래와 같다. 각 출력($o_i$)은 모든 입력($x_j$)을 사용해서 계산되기 때문에 **fully connected layer** 이다.
  
  ![Unitiled](https://ko.d2l.ai/_images/softmaxreg.svg){: .align-center}*Softmax 회귀의 단층 뉴럴 네트워크*
  - 이는 $\mathbf{o} = \mathbf{W}\mathbf{x} + \mathbf{b}$ 로 표현될 수 있다.

## Logistic Regression
- 예컨대 여자/남자를 분류하거나 참/거짓을 분류하는 문제는 $y \in \lbrace 0, 1 \rbrace$ 이라고 할 수 있다. 이러한 이진 분류는 Logistic Regression 을 사용한다.
- Logistic Regression 도 구성 자체는 Linear Regression 과 크게 다르지 않다.
![Untitled](/assets/images/DL_basic/logistic.png){: .align-center}*출처: https://kh-kim.github.io/*
- 위 그림에서 보이는 $\sigma$ 는 **Sigmoid** 함수를 뜻한다. 즉 활성화 함수로 Sigmoid(Logistic) function 을 사용한다. 활성화 함수는 이 [포스트](https://bkkhyunn.github.io/deep_learning/deep_learning_4/)에서 자세히 정리한다.
  
  💡 Linear Regression 에서는 활성화 함수를 쓰지 않았을까?<br>
  활성화 함수는 입력 신호의 총합을 출력 신호로 변환하는 함수다. 이를 통해 이전 층(layer)의 결과값을 변환하여 다른 층의 뉴런으로 전달할 수 있다. 그러나 딥러닝에서 일반적으로 이야기하는 활성화 함수는 **미분 가능한 비선형 함수**이다. 세상은 복잡하기 때문에 선형으로만 설명할 수 없고 비선형의 표현이 필요하기 때문에 딥러닝에서 만날 활성화 함수는 비선형 함수이자, 학습을 위해 미분이 가능해야 한다. 그럼에도 불구하고, Linear Regression 도 가장 간단한 활성화 함수인 선형 함수가 쓰였다. 선형 함수는 입력과 동일한 값을 출력한다. 이러한 선형 함수도 입력 신호를 출력 신호로 변환하기 때문에 활성화 함수라고 볼 수 있다.
  {: .notice--warning}

- Sigmoid 함수를 이용하여 **모델의 출력 값의 범위를 0에서 1사이로 고정**할 수 있다. 이 점을 이용해서 참/거짓 등의 이진 분류를 할 수 있다.

![Untitiled](/assets/images/DL_basic/sigmoid.png){: .align-center}*Sigmoid 혹은 Logistic function 그래프. 출처: cs229 lecture note*
- Sigmoid 를 수식으로 보면 다음과 같다.
    <center>$\sigma(x) = \displaystyle\frac{1}{1+e^{-x}}$</center>
- Sigmoid 함수는 +∞로 갈 경우 1, -∞로 갈 경우 0 이다. 또한 0.5를 기준으로 참과 거짓을 판단할 수 있다. 만약 출력 벡터의 어떤 요소의 값이 0.5 보다 같거나 클 경우 해당 요소에서 참을 예측했다고 결론지을 수 있으며, 반대로 0.5보다 작을 경우 거짓을 예측했다고  결론내릴 수 있다.
- Sigmoid 미분은 아래와 같이 유도될 수 있다.
- 이를 인공 신경망에서 입력값과 가중치가 연산되어 들어온다고 보면 다음과 같다.
    <center>$h_\theta(x) = g(\theta^Tx) = \displaystyle\frac{1}{1+e^{-\theta^Tx}}$</center>
- 이제 Logistic Regression 은 이진 분류(binary classification)에서 사용되고 $y \in \lbrace 0, 1 \rbrace$ 이라고 했으므로,
    <center>
- Logistic Regression 의 Loss function(Cost function)은 기존 Linear Regression 의 MSE 와 다른 함수를 사용한다. 선형 회귀와 마찬가지로 손실값을 가중치 파라미터로 미분하게 되면, 손실값이 낮아지는 방향으로 경사하강법을 수행할 수 있다. 아래에서 살펴볼 것이다.

## Softmax Regression
- 분류에서는 선형회귀처럼 $\mathbf{o}$ 와 $\mathbf{y}$ 사이의 차이를 최소화하는 것은 아래와 같은 이유로 적절하지 않다.
  - 확률로 동작하는 것처럼 출력값($o_i$)의 총합이 1이 되지 않을 수 있다.
  - 출력값($o_i$)이 음수가 아닌 값이더라도 더해서 1이 되지 않거나 1이 초과하지 않을 거란 보장이 없다.
- 또한 선형회귀에서 **MLE** 를 사용하는 것처럼 $\epsilon$이 정규성($\mathcal{N}(0, \sigma^2$))을 띄고, $\mathbf{y} = \mathbf{o} + \epsilon$ 으로 생각해볼 수 있다. 이를 **probit model**이라 한다.
![Untitled](/assets/images/DL_basic/probit.png){: .align-center}*Probit Model 출처: https://en.wikipedia.org/wiki/Probit_model*
- 그러나 Probit Model은 다음에 볼 softmax 에 비해 딥러닝에서 잘 작동하지 않고 optimization 도 어렵다고 알려져 있다.
- 따라서 출력값이 음수가 아닌 수이고 출력값의 합이 1이 되도록 하기 위해서, **지수함수**를 사용한다.
  - 지수함수를 사용하면 $o_i$ 가 증가함에 따라 조건부 클래스 확률도 증가한다. 즉 해당 클래스가 할당될 확률도 증가한다는 것을 의미한다.
  - 지수함수는 단조증가함수이며, 지수함수를 통한 확률은 음이 아닌 값이다.
- 지수함수의 결과를 합으로 나누는 Normalization 을 통해 총합이 1이 되고, 이를 수식화한 것이 **softmax function** 이다.
    <center>$\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})\quad \text{where}\quad \hat{y}_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$</center>
- 가장 큰 $o_i$의 클래스가 분류 클래스가 된다.
    <center>$\operatorname*{argmax}_j \hat y_j = \operatorname*{argmax}_j o_j$</center>
- Softmax 는 현대 물리학에서 기체 분자의 에너지 상태에 대한 분포를 모델링하기 위해 사용한 볼츠만 분포 등에서 사용된 기법이다. 열역학 앙상블에서 특정 에너지 상태가 발생할 빈도는 $\text{exp}(-E/kT)$에 비례한다고 한다. 여기서 $E$는 상태의 에너지, $T$는 온도이며, $k$는 볼츠만 상수이다.
- 통계학자들이 통계 시스템에서 "온도"를 높이거나 낮춘다고 말할 때, 낮은 또는 높은 에너지 상태를 선호하도록 $T$를 변경하는 것을 의미한다. 여기서 에너지는 **오차**에 해당한다. 

## Logit, Sigmoid, Softmax 의 관계

## Loss function(Cross Entropy)
- 
### Binary Cross Entropy

### Cross Entropy

## Information Theory
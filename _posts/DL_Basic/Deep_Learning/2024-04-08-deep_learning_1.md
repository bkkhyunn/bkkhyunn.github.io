---
title: "[Deep Learning] 1. Linear Neural Networks for Regression"
excerpt: "선형 회귀(Linear Regression)에 대해 알아보고 이후 Neural Network 로 Linear Regression 을 하는 과정을 알아보자."

categories: "deep_learning"
tags:
    - deep learning
    - Linear Regression
toc: true  
toc_sticky: true
toc_label: "Contents In Page"
author_profile: true
use_math: true

date: 2024-04-08
---

기존의 머신러닝에서 사용되던 Linear Regression 을 자세히 알아보고 아직 비선형성이 추가되지 않은 Neural Network 를 공부해보자. 여기서 등장하는 수학적 개념들은 AI Math 카테고리에서 더 자세히 다룬다.

## 선형 회귀(Linear Regression)
- 회귀 문제는 주어진 데이터 $x$에 해당하는 실제 값으로 주어지는 타겟 $y$ 를 **예측**하는 과제이다. 즉 독립변수 $𝑥$에 대응하는 종속변수 $𝑦$와 가장 비슷한 값 $\hat{y}$을 출력하는 함수 $f^{\ast}(x)$를 찾는 과정이다.
- 즉, 선형 회귀 모델을 통해서 내가 풀고자 하는 어떤 함수 $f(x)$를 근사하는 함수 $f^{\ast}(x)$를 찾는 것이다.
- Linear Regression 은 "예측하고자 하는 종속변수 y와 독립변수 x 간에 **선형성**을 만족한다 는 가정"을 따른다. 
- 이는 결과 값이 이미지 **분류**와 같이 과일의 종류를 예측하는 **이산적인(discrete) 분류 문제(classification)**와는 다르다.
- $\hat{y} = f(x) \approx y$<br>
  $\hat{y} = w_1x_1+w_2x_2+\cdots+w_Dx_D+b=w^Tx+b$
- 위 식에서 독립변수 $x=(x_1,x_2,\cdots,x_D)$는 𝐷차원 벡터다. 가중치 벡터 $w=(w_1,\cdots,w_D)$와 편향(bias) $b$는 함수 𝑓(𝑥)의 계수(coefficient)이자 이 선형회귀모형의 **모수(parameter)**라고 한다.
- 회귀 문제는 현실에서 많이 보이는 문제이다. 예를 들면, 주택 가격, 기온, 판매량 등과 같은 **연속된 값을 예측**하는 문제들을 들 수 있다.
  - 간단한 예로 집의 면적과 지어진 후 몇 년이 되었는지를 입력으로 사용해서 주택 가격을 예측하는 문제가 있다.
  - $price = w_{area}\cdot area + w_{age}\cdot age + b$<br>
    $\Rightarrow$ $\hat{y} = w_1x_1+w_2x_2+\cdots+w_Dx_D+b=w^Tx+b$
  - 각 데이터 $x_i$와 이에 대한 label 값인 $y$를 추정해서 연관시켜주는 가중치(weight)벡터 $w$와 bias $b$를 찾는다.
- $x$ 와 ​$y$의 관계가 **선형적**이라고 가정하는 것은 상당히 합리적이지만, 오류가 발생할 수도 있다. 예를 들면, 주택 가격은 일반적으로 하락하지만, 오래될수록 가치가 더해지는 오래된 역사적인 주택의 경우는 해당되지 않을 수 있다.
- 파라미터​ $w$를 찾기 위해서는 두가지가 더 필요하다. 1) 하나는, 현재 모델의 성능, 품질을 **측정**하는 방법과 2) 두번째는 성능, 품질을 향상시킬 수 있는 방법 즉 **학습**이다.
- 결국 우리가 하고자 하는 것은 **모델의 예측과 실제값 사이의 차이를 최소화**하는 **모델 파라미터 $w$와 $b$**를 **찾는 것**이다. 이를 통해 풀고자 하는 어떤 함수 $f(x)$를 근사하는 함수 $f^{\ast}(x)$를 찾을 수 있는 것이다.

## Loss function(Mean Squared Error)
- 모델 학습을 위해서 모델의 예측과 실제 사이의 **오차(거리)**를 측정해야 한다. 오차는 0 또는 양수값이고 값이 작을수록 오차가 적음을 의미한다.
- Linear Regression 에서 가장 일반적인 Loss func 은 **squared error** 다.
- $i$번째 샘플에 대한 loss 계산은 $i^{(i)}(\mathbf{w}, b)=\frac{1}{2}(\hat{y}^{(i)}-y^{(i)})^2$ 로 나타낼 수 있다.
![Untitled](https://d2l.ai/_images/fit-linreg.svg){: .align-center}*출처: Dive into Deep Learning* 
- 수식에 곱해진 1/2 상수값은 2차원인 loss(squared error)를 미분했을 때 값이 1이되게 만들어서 조금 더 간단하게 수식을 만들기 위해서 사용된 값이다. 또한 상수를 나누는 것은 모델의 성능에 영향이 거의 없다.
- 데이터는 이미 구했거나 주어져있기 때문에, loss func 은 모델 파라미터($w, b$)에만 의존한다.
- squared loss 는 제곱 형식이기 때문에, 큰 loss 를 더 크게 반영하여 학습에 도움이 되지만 이는 이상치에 큰 영향을 받을 수 있다는 것을 의미한다. (d2l 에서는 양날의 검이라 표현한다.)
- 전체 데이터셋에 대해서 모델의 품질을 측정하기 위해서 학습셋에 대한 loss의 평균값을 사용할 수 있다. 이는 Mean Squared Error 즉 **MSE** 이다.
  - $L(\mathbf{w}, b) = \frac{1}{n}\displaystyle\sum_{i=1}^nl^{(i)}(\mathbf{w},b) = \frac{1}{n}\displaystyle\sum_{i=1}^n\frac{1}{2}(\mathbf{w}^T\mathbf{x}^{(i)}+b-y^{(i)})^2$
  - 이 평균 squared loss(MSE)를 최소화하는 모델 파라미터 $\mathbf{w}^\ast$와 $b^\ast$ 를 찾는 것이 모델을 학습시키는 과정이다.
  - $\mathbf{w}^\ast, b^\ast = \underset{\mathbf{w}, b}{\mathrm{argmin}} \; L(\mathbf{w}, b)$

## 최적화 알고리즘(Minibatch SGD)
- 앞서 정의한 Linear Regression 모델과 loss 함수의 경우, loss를 최소화하는 방법은 **역행렬**을 사용해서 명확한 수식으로 표현할 수 있다.
  - $\parallel \mathbf{y} - \mathbf{Xw} \parallel ^2$ 를 최소화하기만 되고, $\mathbf{X}$는 feature들끼리 선형 의존성이 없기 때문에 loss surface 상에 loss를 최소화할 수 있는 점이 한 개만 존재한다.
  - $\mathbf{w}$ 로 $\parallel \mathbf{y} - \mathbf{Xw} \parallel ^2$ 을 미분하여 0이 되는 지점을 찾으면 된다. 그렇게 되면 아래의 식이 도출된다.
  - $\mathbf{w}^\ast = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$
- 그러나 이 수식은 다양하고 **Multi Layer Perceptron** 이나 **비선형** layer가 있으면 적용할 수 없는 한계가 있다. 따라서 대부분의 딥러닝 모델은 이러한 **역행렬을 이용한 분석적 방법을 적용할 수 없다**.
- 딥러닝에서는 모델의 파라미터를 어떤 최적화 알고리즘을 사용해서 점진적으로 업데이트하면서 loss를 줄여나가는데, 이것이 바로 **gradient descent(경사하강법)**이다.
- 이 때, 매 iteration 마다 전체 데이터셋에 대한 loss func을 계산하고 **loss func 을 $\mathbf{w}$로** **편미분**한 값을 활용하여 gradient descent 를 거쳐 파라미터를 업데이트하는 것은 매우 느리다(정확한 값을 구할 수는 있지만).
- 또한 이론적으로 gradient descent는 미분가능하고 볼록(convex)한 함수에 대해서 적절한 학습률과 학습횟수를 선택했을 때 수렴이 보장되어 있지만, 딥러닝과 같은 비선형회귀의 문제인 경우 많은 경우에 목적식(loss func)이 볼록하지 않을 수 있으므로 수렴이 항상 보장되지는 않는다.
![Untitled](/assets/images/DL_basic/nonconvex.png){: .align-center}*non-convex 목적식* 
- 이를 해결하기 위해 **SGD(Stochastic Gradient Descent)** 를 활용하면 된다.
  - SGD 는 한번의 iteration 에 single 혹은 일부의 sample 을 이용하여 가중치를 업데이트한다.(**one observation one update step**)
  - 볼록이 아닌(non-convex) 목적식은 SGD 를 통해 최적화할 수 있다. 만능은 아니지만 딥러닝의 경우 SGD 가 일반적인 GD 보다 실증적으로 더 낫다고 검증되었다.
- 그러나 SGD 에도 단점이 있는데, 1) 프로세스 측면에서 sample 을 하나씩 처리하는 것이 모든 sample 을 한번에 처리하는 것보다 느릴수도 있다는 점과 2) Batch Normalization 과 같은 layer 는 한 개 이상의 sample 이 있어야 잘 동작한다는 점, 3) gradient 의 추정값이 너무 noise 해진다는 점이다.
- 이러한 이유들을 종합하여, full-batch 와 one single sample 을 이용하는 것이 아닌 **mini-batch** 를 사용한다. 이 때 mini-batch 의 size 는 dataset 의 크기, memory, workers 등에 따라 다르게 설정할 수 있는 **하이퍼 파라미터**이다. 일반적으로 32에서 256 사이의 2의 제곱수로 시작하는 것이 좋다고 알려져 있다.
- 이를 **minibatch stochastic gradient descent** 라고 한다. 딥러닝에서 일반적으로 쓰이는 방식이다.
![Untitled](https://kh-kim.github.io/nlp_with_deep_learning_blog/assets/images/1-07/01-linear_regression_overview.png)*출처: https://kh-kim.github.io <br> 위 그림에서는 minibatch 를 사용한 것은 아니지만 이러한 방식으로 모델이 학습된다.*{: .align-center}<br>
    <center>$(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{\mid\mathcal{B}\mid}\displaystyle\sum_{i \in \mathcal{B}} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b)$</center>
  - 아래의 순서로 사용된다.<br>
  1. (일반적으로)난수를 이용해서 모델 파라미터를 초기화한다.<br>
  2. 학습 데이터에서 batch size($\mathcal{B}$) 만큼의 sample 들을 균일한 방식으로 뽑아서 minibatch 를 구성한다. minibatch 를 구성하는 sample 은 다양한 방식을 사용할 수 있다. 클래스 간 균형을 위해 Startified Sampling 등을 사용할 수 있다.<br>
  3. minibatch 의 데이터를 모델에 통과(feed-forward)시키고 나온 출력 $\mathbf{Xw}$ 과 실제값 $\mathbf{y}$ 을 비교한 뒤 평균 loss 값을 얻는다. 이를 **모델 파라미터($\mathbf{w}$)로 편미분**한다.<br>
  4. 이 결과와 미리 정의된 **학습률**(learning rate = step size, $\eta>0$)를 곱해서 **loss 값이 최소화되는 방향으로 파라미터를 update** 한다.
- 여기서 중요한 점은 **batch size**와 **learnig rate**는 모델 학습을 통해서 찾아지는 값이 아니라 직접 선택을 해야하는 **hyper-parameters** 이다.
- **minibatch 는 batch size 에 따라 확률적으로 선택하므로 매 iteration 마다 목적식 모양이 바뀌게 된다**. 즉 매번 다른 minibatch 를 사용하기 때문에 곡선 모양이 바뀌는 것이다. 또한 SGD는 non-convex 한 목적식에서도 사용가능하기 때문에 학습에 더 효율적이다.
![Untitled](/assets/images/DL_basic/minibatchsgd.png){: .align-center}*좌: 경사하강법(GD), 가운데: 확률적 경사하강법(SGD), 우: GD vs. minibatch SGD* 

## Inference
- 모델 학습이 끝나면 모델 파라미터 $(\mathbf{w}, b)$에 해당하는 값 $\hat{\mathbf{w}}$ 와 $\hat{b}$ 을 저장한다. 이 때 학습을 통해서 loss 함수를 최소화 시키는 최적의 값 $\mathbf{w}^{\ast}, b^{\ast}$를 정확하게 구할 필요는 없다. 이를 정확히 아는 것부터 매우 어렵고, **학습을 통해서 이 최적의 값에 근접하는 값을 찾는 것으로 충분**하다.
- 이후, 학습된 선형 회귀 모델 $\hat{\mathbf{w}}^Tx + \hat{b}$을 이용해서 학습 데이터셋에 없는 데이터에 대해 결과값을 추정한다. 이를 “모델 예측(prediction)” 또는 “**모델 추론(inference)**” 라고 하고, **validation** 혹은 **test** 단계에서 진행한다.

## Vectorization
- 딥러닝 연산을 할 때 보통 **numpy** 를 활용한 연산을 진행한다. torch 나 tensorflow 역시 numpy 를 base 로 한다.
- numpy 는 일반 파이썬의 for 문 보다 훨씬 빠른데, 그 이유는 numpy 는 각 element 에 대해 for 문을 수행하는 것이 아니라 **전체 array 에 대해 vectorization 연산을 수행**하기 때문이다. 따라서 torch 의 연산도 빠르다.
```python
n = 10000
a = torch.ones(n)
b = torch.ones(n)
c = torch.zeros(n)
t = time.time()
for i in range(n):
    c[i] = a[i] + b[i]
f'for loop: {time.time() - t:.5f} sec'
t = time.time()
d = a + b
f'vectorization {time.time() - t:.5f} sec'
```
```python
'for loop: 0.17802 sec'
'vectorization 0.00036 sec'
```
- vectorization은 하드웨어 상의 SIMD (Single Instruction Multiple Data) 동작과 관련있다.
- 최신 64비트 프로세서들은 정수에 대해 수치 연산을 수행하는 64 bit register 16개를 포함하고 있다. 이러한 register들은 한번에 하나의 값만 가질 수 있기 때문에 scalar register라고도 불린다.
- 이러한 칩들은 256 bit register 16개를 추가적으로 포함하도록 발전했다. register 용량이 커져 64 비트값 4개, 32 비트값 8개 등을 동시에 저장할 수 있으니 vector register라 불린다.
- 따라서 **register에 올라가 있는 여러 값들을 한번에 수치 연산할 수 있는 instruction이 가능해지게 된 것**이다. 즉, single instruction에 multiple data를 처리할 수 있게 된 것.
- 결과적으로 vectorization은 scalar register 대신 vector register를 이용하여 수치 연산을 더 빠르게 하는 과정을 의미한다. scipy나 numpy 등이 vectorization을 활용하는 대표적인 라이브러리이고, numpy의 built-in 함수를 쓰는 것이 vectorization 관련해서 최적화가 되어 있으니 훨씬 빠르다.
- **따라서 minibatch 를 활용한 학습이 빠르게 진행될 수 있는 것이다.**

## 정규분포(Normal Dist)와 Squared Loss
💡 딥러닝은 **확률론 기반의 머신러닝 이론**에 기반을 두고 있다. 위에서 본 loss func 의 작동원리는 데이터 공간을 통계적으로 해석해서 유도하게 된다. 이렇게 유도된 loss func 을 가지고 학습을 통해서 예측이 틀릴 위험(risk)을 최소화하도록, 즉 **데이터에서 관찰되는 분포**와 **모델 예측의 분포의 차이를 최소화**하는 방향으로 유도한다. AI Math 포스트에서 자세히 다루겠지만, **loss 를 최소화하기 위해서는 측정하는 방법을 알아야** 하고, 이 측정하는 방법을 통계학에서 제공하기 때문에 머신러닝, 딥러닝을 이해하려면 확률론의 기본 개념을 알아야 한다. 또 다른 이유로 딥러닝은 내가 풀고자 하는 함수의 파라미터를 모사하는 것이기도 하지만 **확률 분포를 모사하는 것**으로 해석이 가능하다. 따라서 확률론은 중요하다.
{: .notice--warning}

- 확률분포는 데이터공간의 초상화로서, 데이터를 바라보고 해석하는데 굉장히 중요한 도구이다. 이 중 조건부확률 $\mathbf{P}(y\|\mathbf{x})$ 는 입력변수 $\mathbf{x}$에 대해 정답이 $y$일 확률을 의미한다. 단, 선형회귀와 같은 연속확률분포에서는 확률이 아니라 **밀도**로 해석한다.
- 선형회귀에서는 **조건부기대값 $\mathbb{E}[y\|\mathbf{x}]$** 을 추정하는데, 그 이유는 **조건부기대값**을 추정하는 것이 선형회귀의 목적식인 **Mean Squared Loss(L2 norm 의 기대값)를 최소화하는 함수(내가 찾고자 하는 함수)와 일치**하기 때문이다. 이는 수학적으로 증명되어 있고 이 [블로그](https://blog.naver.com/juhy9212/220961666263)에서 증명을 확인할 수 있다.
- **기대값**은 데이터를 대표하는 통계량으로 사용할 수 있으면서, 모델링하고자 하는 확률분포에서 계산하고 싶은 다른 통계적 범함수를 사용하는데 이용되는 도구이다. 평균과 동일한 개념으로 사용할 수 있지만 머신러닝, 딥러닝에서는 더 폭넓게 사용된다. 목적으로 하는 함수가 있을 때 그 함수의 기대값을 계산하는 것을 통해서 데이터를 해석할 때 여러 방면에서 사용할 수 있다.
- 딥러닝에서도 MLP 를 사용하여 데이터로부터 **특징패턴** $\phi(\mathbf{x})$를 추출하고 **가중치행렬** $\mathbf{W}$ 를 통해 조건부확률을 계산하거나 혹은 조건부기대값을 추정하는 식으로 학습을 하게 된다.
- 다시 돌아와서, **정규분포**와 **Mean Squared loss(MSE) 를 활용하는 Linear Regression** 은 관련이 깊다.
- squared loss는 간단한 편미분이라는 특징으로 gradient가 예측값과 실제값의 차이로 계산된다.
  - $\text{squared loss}\; l(y,\hat{y})=\frac{1}{2}(y-\hat{y})^2 \rightarrow \partial_{\hat{y}}\;l(y-\hat{y})=(\hat{y}-y)$
  - MSE 는 이를 **평균** 취한 것이다.
- 평균이 $\mu$, 분산이 $\sigma^2$ 일 때 정규분포는 다음과 같이 나타낸다.<br>
<center>$p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (x - \mu)^2\right)$</center><br>

![Untitled](https://d2l.ai/_images/output_linear-regression_d0729f_78_0.svg){: .align-center}
- 평균을 변경하면 $x$ 축을 따라 이동하는 것과 일치하고 분산을 늘리면 분포가 확산되어 피크(높낮이)가 낮아진다.
- 통계적 모델링은 적절한 가정 위에서 확률분포를 추정(inference)하는 것이 목표이다. 그러나 유한한 개수의 데이터만 관찰해서 모집단의 분포를 정확하게 알아낸다는 것은 불가능하므로, 근사적으로 확률분포를 추정할 수 밖에 없다.
  - 데이터가 특정 확률분포를 따른다고 선험적으로(a priori) 가정한 후 그 **분포를 결정하는 모수(parameter)**를 추정하는 방법을 모수적(parametric) 방법론이라 한다.
  - 특정 확률분포를 가정하지 않고 데이터에 따라 모델의 구조 및 모수의 개수가 유연하게 바뀌면 비모수(nonparametric) 방법론이라 부른다. 머신러닝의 많은 방법론은 비모수 방법론에 속한다.
- Squared Error 를 적용한 선형 회귀모델에서 중요한 가정은 **관측값($\hat{y}$)이 noise가 있는 정규 분포 $\mathcal{N}(0, \sigma^2)$에서 얻어지고, 이 noise 들은 데이터에 더해진다는 것**이다. 즉, 각 $\mathbf{x}$ 값 별로 모집단 내 $y$ 값들이 정규분포를 따르고 있다는 것이고, 선형 회귀모델은 정규 분포를 모사하고 있다는 것이다.
  
    <center>$y = \mathbf{w}^\top \mathbf{x} + b + \epsilon \text{ where } \epsilon \sim \mathcal{N}(0, \sigma^2)$</center>

  💡 왜 정규분포를 가정할까? 예측을 할 때는 "내 예측값이 실제값과 정확히 예측한다!" 라고 말할 수 없다. 언제나 예외는 존재하는 법이고, 정확히 예측할 수 있다면 머신러닝과 딥러닝이 필요없을 것. 따라서 "실제값은 내 예측값일 확률이 가장 높지만, 아닐 수도 있다!" 라고 말하는 것이 현명하다. 여기서 "아닐 수도 있다" 는 것은 바로 유한한 개수의 데이터만 관찰해서 모집단의 분포를 정확하게 알아내는 것이 불가능하여 **근사적으로 확률분포를 추정할 수 밖에 없기 때문**이다. 이는 근본적으로 발생하는 불확실성이다. 이 때, **실제값은 내가 모르기 때문에 Random Distribution 인데, 내가 예측한 값을 평균으로 하는 정규분포(가우시안분포)를 따른다고 볼 수 있다**고 말하는 것이 앞서 한 현명한 말과 비슷한 의미가 된다. 왜냐하면 정규분포는 평균에서 확률밀도가 가장 높기 때문이다. 또 이를 다시 말하면 **"실제값의 분포는 나의 예측값을 평균으로 하고 $\sigma^2$를 분산으로 하는 정규분포를 따른다!"** 라고 말할 수 있는 것이다.
  {: .notice--warning}

- **모수를 추정**할 때는, **가정된 확률분포에 대응하는 모수의 값으로 관측값이 설명될 수 있는 가능성**을 가지고 추정한다. 이를 **가능도 추정법**(likelihood estimation)이라고 한다. 즉 가능도 함수는 모수 $\theta$를 따르는 분포가 $\mathbf{x}$를 관찰할 가능성을 뜻한다.
  - 확률밀도함수에서는 모수 $θ$가 이미 알고 있는 상수 벡터이고 $x$가 변수(벡터)이다. 하지만 모수 추정 문제에서는 이름이 내포하는 것과 같이 $x$, 즉, 어떤 분포에 대해 이미 실현된 표본값은 알고 있지만 역으로 모수 $θ$를 모르고 있는 상황이며 이 모수를 추정하는 것이 관심사이다.
  - 이 때는 확률밀도함수에서와는 반대로 $x$를 이미 알고 있는 상수 벡터로 놓고 $θ$를 변수(벡터)로 생각한다. 물론 함수의 값 자체는 여전히 동일하게, 주어진 $x$에 대해 나올 수 있는 확률밀도이다. 달라진 것은 원래의 확률밀도함수가 $x$를 변수로 하는 함수였다면 가능도함수는 $\theta$를 변수로 하는 함수가 된 것이다.
  - 이렇게 확률밀도함수에서 모수를 변수로 보는 경우에 이 함수를 **가능도함수(likelihood function)**라고 한다. 같은 함수를 확률밀도함수로 보면 $p(x\|θ)$로 표기하지만 가능도함수로 보면 $L(θ\|x)$ 기호로 표기한다. 동일한 대상이지만 바라보는 관점의 차이를 반영하는 표기인 것이다.

  💡 딥러닝 입장에서, **가능도**는 파라미터(\$w$)가 주어졌을 때 예측값이 등장할 확률이다. 즉 우리는 맨처음 난수로 초기화하거나 학습을 진행하면서 업데이트된 파라미터 $w$를 가지고, 가능한 모든 데이터 $x$ 에 대하여 예측값 $y(x\|w)$이 실제값에 최대한 가깝게 위치하도록 하는 것이다. 이 때 파라미터 $w$ 는 $y(x\|w)$가 가지는 파라미터이고 가중치와 편향을 모두 포함한다.
  {: .notice--warning}

- 이제 이 선형회귀모델이 따른다고 가정한 정규분포를 대상으로 모수를 추정해보자.
- $\mathbf{x}$가 주어졌을 때 $y$ 의 가능도 함수는 아래와 같이 표현될 수 있다. 여기서 모수, 즉 파라미터는 $\mathbf{w}$ 와 $b$ 가 된다.
<center>$P(y \mid \mathbf{x}) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (y - \mathbf{w}^\top \mathbf{x} - b)^2\right)$</center>

- 일반적으로 모수의 추정을 위해 확보하고 있는 확률변수 표본의 수가 하나가 아니라 여러개 $x= (x_1,x_2,⋯,x_n)$이므로, **가능도함수 또한 복수의 표본값에 대한 결합확률밀도가 된다.** 또한 표본 데이터 $x_1,x_2,⋯,x_n$는 같은 확률분포에서 나온 독립적인 값들이므로$(X_1=x_1, \cdots, X_n=x_n)$ 결합확률밀도함수는 독립사건의 확률 계산에 의해 다음처럼 곱으로 표현된다.
<center>$P(\mathbf y \mid \mathbf X) = \prod_{i=1}^{n} p(y^{(i)}|\mathbf{x}^{(i)})$</center>

- 위 식의 값(가능도)이 가장 커지는 **$\mathbf{w}$** 및 **$b$**가 전체 데이터셋의 **가능도를 최대화**하는 값이고 **최적의 값**이 된다. 가능도가 최대값이 될 때, 가정한 확률분포에 대응하는 모수의 값이 관측값을 설명할 수 있을 가능성이 가장 높기 때문이다. 따라서 이 방식을 **Maximum Likelihood Estimation (MLE)** 라고 부른다.
- 가능도의 최대값은 로그를 취한 후 최대값을 찾는다. 또한 최적화는 최대화보다는 최소화로 표현되는 경우가 더 많기 때문에(경사하강법은 최소값을 찾는다), **음의 로그 가능도** $-\log P(\mathbf y \mid \mathbf X)$를 최소화한다.
  - 로그가능도를 최적화하는 모수 $\theta$ 는 가능도를 최적화하는 MLE 가 된다.
  - 로그를 사용하는 이유는, 데이터의 숫자가 적으면 상관없지만 만일 데이터의 숫자가 수억 단위가 된다면 컴퓨터의 정확도로는 기존의 가능도를 계산하는 것은 불가능하다.
  - 또한 데이터가 독립일 경우, 로그를 사용하면 가능도의 곱셈을 로그가능도의 **덧셈**으로 바꿀 수 있기 때문에 컴퓨터로 연산이 가능해진다. 더욱이 가능도가 최대면 로그가능도도 최대이기 때문에 문제가 발생하지 않는다.
  - **경사하강법**으로 가능도를 최적화할 때 미분 연산을 사용하게 되는데, 로그가능도를 사용하면 연산량을 $O(n^2)$에서 $O(n)$으로 줄여준다.
  - 대게의 손실함수의 경우 경사하강법을 사용하므로 음의 로그가능도(negative log-likelihood)를 최적화하게 된다.
<center>$-\log P(\mathbf y \mid \mathbf X) = \sum_{i=1}^n \frac{1}{2} \log(2 \pi \sigma^2) + \frac{1}{2 \sigma^2} \left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b\right)^2$</center>

- 위 공식을 활용하여 $-\log P(\mathbf y \mid \mathbf X)$를 최소화 해보자.
  - $\sigma$와 $\pi$ 는 상수값이라 로그 가능도를 최대화하는 $\mathbf{w}$ 와 $b$ 값에 영향을 주지 않는다.
  - 음의 부호가 붙어 최대화를 위해서는 최소화를 해야하므로, $\left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b\right)^2$ 이 식을 최소화시키는 $\mathbf{w}$ 와 $b$ 를 찾으면 된다.
- 잘 보면, 이는 **Squared Error(L2 Loss)** 이다. 즉, **Gaussian noise를 갖는 선형 모델의 likelihood를 최대화하는 것은 squared loss를 적용한 선형 회귀와 동일한 문제로 정의**될 수 있는 것이다.

  💡 이렇게 MLE를 이용하면 Regression task에서 L2 Loss(MSE)를 쓰는 이유를 증명해 낼 수 있다. 반대로 생각하면, 우리가 앞으로 **Deep Learning 에서 L2 Loss를 이용한다는 것은 주어진 Data로부터 얻은 Likelihood를 최대화시키겠다는 뜻으로 해석할 수 있을 것**이다. 왜냐하면 위에서 본 것처럼 **L2 Loss를 최소화 시키는 일은 Likelihood를 최대화 시키는 일이기 때문**이다! 그러면, 딥러닝에서 목표는 내가 풀고자 하는 함수를 잘 근사하는 함수를 찾아내는 것이고, 이를 위해 loss 를 측정하는데 Regression 에서는 이 loss 를 최소화시키는 가중치가 실제값에 가장 높은 확률로 맞는 예측값을 낼 수 있도록 한다는 것이다! 참고로 Classification 문제에서는 Bernoulli Distribution을 이용하면 비슷한 방법으로 Cross Entropy Error를 유도할 수 있다.
  {: .notice--warning}

- 이 로그가능도를 **최소화하는 지점은 미분계수가 0이 되는 지점**을 찾으면 된다. 찾고자 하는 파라미터에 대하여 로그를 취한 로그 가능도 함수를 편미분하고 그 값이 0이 되도록 하는 파라미터를 찾는 과정을 통해 가능도 함수를 최대화시켜줄 수 있는 파라미터를 찾을 수 있다.

## 이제 Neural Network 로
- Neural Network 에서는 Linear Regression 을 어떻게 다룰까?
- Neural Network 구조로 선형 회귀를 표현해보면, 아래 그림과 같이 Neural Network 다이어그램을 이용해서 선형 회귀 모델을 도식화 할 수 있다.

![Untitled](https://ko.d2l.ai/_images/singleneuron.svg){: .align-center}

- 위 Neural Network 에서 입력값의 개수를 feature dimension 이라고 부르기도 한다. 위 경우에 입력값의 개수는 $d$ 이고, 선형 회귀의 결과인 출력값의 개수는 1 이다. input layer 에는 어떤 비선형이나 어떤 계산이 적용되지 않기 때문에 이 네트워크의 총 layer 의 개수는 1이다. 모든 입력들이 모든 출력(이 경우는 한개의 출력)과 연결되어 있기 때문에, 이 layer 는 fully connected layer 또는 dense layer라고 불린다.

## Reference
https://d2l.ai/chapter_linear-regression/linear-regression.html<br>
https://hongl.tistory.com/282<br>
https://hyeongminlee.github.io/post/bnn002_mle_map
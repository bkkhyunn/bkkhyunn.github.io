---
title: "[TroubleShooting-Pytorch] Deep Learning for HPC"
excerpt: "HPC(High Performance Computing)을 위해 Pytorch 는 무엇을 적용하고 있는지 알아보자."

categories: "troubleshooting-pytorch"
tags:
    - pytorch
    - GPU
    - HPC
toc: true  
toc_sticky: true
toc_label: "Contents In Page"
author_profile: true
use_math: true
date: 2024-04-22
---

## HPC
- HPC 란 High Performane Computing 의 약자로, 많은 뜻이 담겨 있지만 딥러닝에서 중요한 의미를 가진 것은 **복잡한 계산을 병렬로 고속 처리하는 것**이다.
- 하드웨어적 개념들이 많이 들어가 있다.
- 인공지능 뿐 아니라 하드웨어까지 이해하는 사람이 되어야지만 좋은 딥러닝 모델을 효율적으로 운용할 수 있다! 역시 가장 먼저 떠오르는 것이 바로 GPU 이다.
- **우리는 왜 딥러닝을 하는데 하드웨어, 특히 GPU 가 왜 필요할까?**
- 딥러닝의 근간은 인간의 뉴런을 모방한 인공신경망이다. 인공신경망에서 하는 연산을 추상화하면, 들어오는 input 과 weight 들은 모두 **벡터**가 된다. input 과 weight 에 대해 내적을 하고 합을 구한 뒤 activation func 을 거쳐 output 이 만들어진다.
![Untitled](/assets/images/TroubleShooting/neuron.png){: .align-center}
- 이 뉴런들은 하나만 있는 것이 아니고 **병렬로 존재**하고, 연산을 병렬적으로 무수히 많이 할 수 있는 특징을 가지고 있다!
- 그렇다면 내적도 병렬로 이루어질 것이고 내적을 병렬화 한 것은 행렬 곱셈과 같기 때문에, 딥러닝에서 가장 근간이 되는 연산은 input 과 weight 에 대한 **행렬 곱셈**이 된다!
- 또한 행렬 연산 이외에 activation func 같은 경우에도 하나의 뉴런만이 아니라 모든 존재하는 뉴런에 대해서 병렬적으로 적용하기 때문에 **벡터 연산**이라고 볼 수 있다.
- 따라서 **딥러닝에서 근간이 되는 연산은 행렬 연산과 벡터 연산**이고, 이 두 연산 모두 **병렬적**으로 적용할 수 있는 연산이 된다. 이처럼 병렬적으로 처리하는 것이 매우 용이하기 때문에 선형대수를 적용할 수 있는 것이다.
- convolution 연산도 행렬로 표현할 수 있다! 즉 행렬연산으로 표현이 가능하다. 오른 대각선으로 값이 다 똑같은 **Hankel matrix**로 표현될 수 있는데, 이를 통해 행렬 연산을 진행할 수도 증명할 수도 있다.
- 정리하면,
  - 딥러닝의 연산 대부분은 행렬곱셈 연산이다.
  - 이외의 다른 연산은 대부분 병렬화 가능한 벡터 연산이다.
  - 마지막으로, 이러한 딥러닝 연산의 병렬적인 특성으로 인해서 특화된 하드웨어를 활용하는 것이 효율적인 솔루션이 될 수 있다. 그것이 바로 GPU 이다.

## Parallel Hardware
- Flynn's Taxonomy
    ![Untitled](/assets/images/TroubleShooting/flynn.png){: .align-center}
  - Flynn 이 만든 분류방법. 일반적인 프로그래밍 방법을 SSID(Single Instruction Single Data)라고 부른다. 명령 하나를 내리면 데이터 하나에 대해서 명령이 처리됨을 의미한다. 일반적인 프로그래밍이나 개발 방식은 SSID 방식이다.
  - 두 벡터의 덧셈을 for loop 으로 할 수 있지만 한 번에 덧셈을 하는 경우 SIMD(Single Instruction Multiple Data)에 해당한다. 이는 명령을 한 번 내리면 여러 데이터가 동시에 그 명령을 수행하는 방식이다. 실제로 하드웨어에 구현이 되어 있다.
  - 여기서 더 나아가서 MIMD(Multiple Instruction Multiple Data) 방식이 있다. 여러번의 거쳐서 다른 데이터에 다른 명령을 내리는 것은 시간이 오래걸리니, 한번에 여러 명령을 여러 데이터에 내리는 방식이다.
  - MISD 는 실제로 쓰는 일이 없다.
- MIMD 방식이 우리가 다가가려고 하는 부분이다!
- CPU 에 SIMD 까지 다 존재하고 있다. 벡터 연산에 많이 활용되고 있다.

## CPU vs. GPU
- GPU 와 CPU 는 근본적으로 다른 것을 목표로 하고 있다. 따라서 완전히 다른 구조와 다른 trade-off 를 가진다.
![Untitled](/assets/images/TroubleShooting/cpugpu.png){: .align-center}
- **CPU** 의 목적은 빠르게 하는 것도 중요하지만 많은 명령을 모두 다 해내는 능력이 더 중요하다. 이에 따라 복잡하고 논리적인 구조를 가지고 있고 명령이 들어올 때 다양한 제어 등의 프로세스를 거치게 된다. 따라서 실제로 연산하는 부분은 상대적으로 작아질 수 있다.
- **GPU** 는 원래 게임에서 그래픽을 빠르게 연산하기 위해 만들어진 장비기 때문에, 그래픽 혹은 병렬 연산이라는 한가지 과제에만 집중되어 있다. 따라서 상대적으로 논리 혹은 제어 등의 시스템에 사용되는 비중을 줄이고 연산을 하는 비중에 훨씬 더 집중했다. 그래서 연산기의 숫자가 훨씬 더 많다.
- 개별적인 연산기의 힘은 CPU 가 더 강하다. 대신에 많은 숫자로 인해서 훨씬 더 강한 힘을 내는 것은 GPU 이다.
- NVIDIA 의 경우 이러한 GPU 시스템을 보고 SIMT(Single Instruction Multiple Thread) 라고 부르고 있다. MIMD 와 유사하지만 약간의 명령 내릴 때 제약이 있는 형태인 것이다.
- GPU 로 병렬적인 연산을 효율적으로 진행할 수 있다. 그러나 CPU 에서는 하드웨어가 여러 제어를 도와주는 반면, GPU 에서는 하드웨어 제어가 상대적으로 적어서 개발자가 이를 담당해야 한다.

## Deep Learning by GPU
- 다시 한번, 딥러닝 연산의 근간은 행렬 곱셈 연산과 벡터 연산이다!
- 이 중에 행렬 연산을 GPU 가 어떻게 할까?
![Untitled](/assets/images/TroubleShooting/gpumatrix.png){: .align-center}
- 위 그림이 **GPU** 에서 **행렬 연산**을 처리하는 근간인 **Tiled Matrix Multiplication** 이다.
- 선형대수에서 Blocked Matrix Multiply 라는 개념이 있다.
  - 행렬 곱셈을 할 때 기본은 row 1개와 column 1개를 가지고 원소별로 곱한 뒤 summation 을 해서 새로운 1칸을 채워나간다.
  - 이 때, 수학적으로 block 단위로 행렬을 쪼개서 block 을 가지고 행렬 곱셈을 해서 결과를 출력하더라도 수학적으로 동일하다.
  - 이 [블로그](https://gaussian37.github.io/math-la-block_matrix_multiplication/)에 잘 정리되어 있다.
- 실제로 컴퓨터의 경우 block 단위로 잘라서 행렬 곱셈을 처리하는 Blocked Matrix Multiply 를 한다.
- 왜 이렇게 block 으로 잘라서 연산하냐하면, 컴퓨터에는 cache 라는 읽기 빠르지만 크기가 작은 메모리가 있다. 이 cache 내에 들어갈 수 있는 정도의 tile(block)을 잘라서 읽어들인 다음에 행렬곱셈을 처리해서 출력을 한다. CUDA 도 이 tiling 을 이용하고 있다.
- 그렇다면 어떻게 병렬화 할 수 있을까?
- 행렬 연산은 한번의 행렬 곱셈에서 사용되는 행과 열을 고려해볼 때 각각의 행렬 곱셈 연산에서 서로 아무런 상호작용이 없다. 최종적으로 summation 이 있지만, 그 전의 연산 중에는 서로 영향을 주지 않는다. Block Matrix Multiply 도 마찬가지다.
- 이처럼 병렬 가능한 특징을 가지고 있기 때문에 행렬 연산을 block 단위로 쪼개서 연산을 진행하면서, 병렬화 처리를 해서 GPU 같은 병렬 하드웨어에서 효율적으로 처리할 수 있다.

## BLAS
- 선형대수를 위한 루틴들을 정리해놓은 표준
- 자주 보게되는 GEMM(Dense 한 행렬곱셈) 연산 등을 정의를 하고 기본적인 operator 를 제공해주는 표준
- Computational Linear Algebra 에서 가장 오래되고 근간이 되는 표준
- 넘파이나 파이토치 같은 라이브러리들은 BLAS 에 담겨있는 구현체들을 API 를 통해서 호출하는 방식으로 적용하고 있다.
- 따라서 매우 효율적으로 행렬연산을 진행할 수 있다.

## Array View
- 우리는 행렬, 텐서를 다룰 때 다차원적인 모양을 생각한다. 그러나 컴퓨터 메모리는 1차원적 구조를 가진다. 그렇다면 1차원적인 메모리를 어떻게 다차원적으로 표현할 수 있을까?
- 이 때 **읽는 방향과 간격을 통해서 표현**할 수 있다.
![Untitled](/assets/images/TroubleShooting/ArrayViews.png){: .align-center}
- 실제 데이터는 1차원적으로 있더라도, 위 그림처럼 읽어서 2차원 행렬로 해석할 수 있는 것이다. 물론 3차원적으로 해석할 수도 있다.
- 행 방향, 열 방향 어느 방향으로 읽을 것인가 생각해볼 수 있다.
  - **Row-major** : Python, CUDA, C++ $\rightarrow$ C 언어에서 처음 나왔기 때문에 C contiguous 방향이라고 부른다.
  - **Column-major** :  $\rightarrow$ MATLAB, R 등의 언어는 수학을 위해 만들어진 프로그램 언어라서 column-major 로 데이터를 읽는다. 이를 보고 F contiguous 라고 한다.
- 읽는 방향을 정의를 함으로써 차원을 정의한다면, 읽는 방향을 바꿔서 transpose 같은 연산이 가능하다.
- 전체 메모리를 읽어서 실제로 메모리 상 transpose 를 하는 것보다, 읽는 방향만 바꿔서 마치 transpose 를 한 것처럼 할 수 있다. 이를 non-contiguous view 라고 한다.
- 차원의 순서를 뒤집거나 읽는 방향을 뒤집거나 하는 연산 처리를 했을 때에는 불연속(non-contiguous)적인 view 를 만들고, 기저에 있는 메모리는 동일하게 있는데 메모리를 읽는 방식을 바꿈으로써 2차원을 3차원 Array 로 바꾼다던지, Transpose 연산을 적용한다던지 할 수 있다.
- 즉 동일한 데이터에 대해서 다른 view 를 제공하는 방식을 쓸 수 있다. 실제로 [pytorch doc](https://pytorch.org/docs/stable/tensor_view.html) 에서도 어떤 view 가 존재하는지에 대해서 정리가 되어 있다.
- slicing 도 마찬가지로 포인터만 제공하면 되기 때문에 view 형태로 제공할 수 있고, 기타등등 많은 tensor operation 들이 실제 메모리를 deep copy 하는 것이 아니라 shallow 하게 포인터만 전달해주는 방식으로 연산을 진행할 수 있다.

## Fundamental Array Concepts with NumPy
- Numpy 에서도 논문을 낸다. 아래 그림은 Numpy 논문에서 발췌한 내용이다.

![Untitled](/assets/images/TroubleShooting/Numpy.png){: .align-center}

- Numpy 의 경우 여러가지 벡터와 행렬 연산을 하기 위한 최적화된 기술이 있다. 이를 모두 **메모리 사용을 최소화하기 위해서 활용**하고 있다. 이를 이해하면 좋다.
- 벡터화, 인덱싱을 view 로 제공해주는 것, 브로드캐스팅을 할 때 더 큰 메모리를 allocation 먼저 하지 않고 최종 output 만 allocation 하는 등 여러가지 모두 **메모리 사용을 최소화하기 위해** 만들어진 것이다!

## Implications for Deep Learning Frameworks
- 메모리 사용을 최소화하는 것은 딥러닝 프레임워크에도 영향을 미쳤다.
- pytorch 의 경우 CNN 을 Channel 이 맨 먼저 위치($C \times H \times W$)한다.
- 왜 이런 방식을 채택했냐면,
  - CNN 의 경우 가장 큰 차원이 activation 의 $H * W * Depth\text{(out_channels)}$ 차원이다. 여기서 depth 는 kernel 의 개수를 뜻한다.
  - 공간 차원(spatial dimension, $H \times W$)이 연산을 처리해야 하는 부분이기 때문에, 이 spatial dimension 이 연속적이어야 한다. 이 때문에 Channel 을 맨 앞에 두고 spatial dimension 을 뒤에 두는 방식이다. 이렇게 되면 **채널 간의 메모리 상의 연속성을 유지**할 수 있게 되는 것이다.
- tensorflow 는 이렇게 하지 않았기 때문에 tensorflow 가 CNN 을 GPU 에서 사용하는데 비효율적이라고 알려져 있다.
- 더 나아가서도 transformer 모델들의 경우도, 실제 내부 API 를 살펴볼 때 Batch, Sequence, Feature 차원으로 되어있다. 왜나하면 Feature 에서 연산이 진행되어야 하기 때문이다. 따라서 Feature 를 연속적으로 읽어야 하기 때문에 가장 안쪽에 위치한 것이다.
- RNN 을 보면 반대로 Sequence, Batch, Feature 순으로 되어 있다. 이는 RNN 의 구조를 살펴보면 이해할 수 있다.
- torch 의 `nn.Linear`의 경우를 보자. 행렬 곱셉의 방식은 A 행렬의 행과 B 행렬의 열을 곱셉을 진행한다. 이 때 메모리 연속성을 고려하여 $y=xA^T+b$ 방식을 취하고 있다. $A$ 에 transpose 를 해준 이유는, torch 와 같은 C 기반 프로그래밍 언어에서는 Row-major 이고 이를 transpose 를 취하면 Column-major 가 된다. 그러면 x 행렬의 행과 A 행렬의 열을 곱하는 것과 같은 효과를 얻을 수 있다.
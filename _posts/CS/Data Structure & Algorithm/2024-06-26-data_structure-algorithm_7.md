---
title: "[Algorithm] 7. Dynamic Programming"
excerpt: "DP 에 대해 알아보자"

categories: "data_structure-algorithm"
tags:
    - algorithm
    - Dynamic Programming
toc: true  
toc_sticky: true
toc_label: "Contents In Page"
author_profile: true
use_math: true

date: 2024-06-26
---

### 다이나믹 프로그래밍(DP)
- 다이나믹 프로그래밍은 메모리를 적절히 사용하여 수행 시간 효율성을 비약적으로 향상시키는 방법이다.
- **이미 계산된 결과(작은 문제)는 별도의 메모리 영역에 저장하여 다시 계산하지 않도록 한다.** 즉 나중에 해당 결과가 필요할 때 메모리 영역에 기록되어 있는 정보를 그대로 사용한다. 한 번 계산해서 해결한 문제는 다시 해결하지 않도록 하는 것이다.
- 컴퓨터 프로그래밍을 할 때는 항상 한 번 해결한 문제는 다시 해결하지 않도록 효율적으로 작성할 수 있는 방법을 고민하면서 프로그램을 작성할 필요가 있다.
- 이러한 점에서 다이나믹 프로그래밍은 완전 탐색을 이용했을 때의 매우 비효율적인 시간 복잡도를 가지는 문제를 다이나믹 프로그래밍을 이용해서 시간 복잡도를 획기적으로 줄일 수 있는 경우가 많다.
- 다이나믹 프로그래밍의 구현은 일반적으로 두가지 방식(top down 과 bottom up)으로 구성된다.
  - top down 방식은 위에서 부터 아래로 내려간다고 해서 하향식, bottom up 방식은 아래에서 위로 올라간다고 해서 상향식이라고 부른다.
- 또한 다이나믹 프로그래밍은 동적 계획법이라고도 부른다.
- 일반적인 프로그래밍 분야에서의 동적(Dynamic)이란 어떤 의미를 가질까?
  - 자료구조에서 동적 할당(Dynamic Allocation)은 프로그램이 실행되는 도중에 실행에 필요한 메모리를 할당하는 기법을 의미한다.
  - 즉 메모리 공간이 필요할 때마다 필요한 만큼의 공간을 프로그램이 실행되는 도중에 할당하는 기법이다.
  - 반면에 다이나믹 프로그래밍에서 다이나믹은 별다른 의미 없이 사용된 단어이다.
- 다이나믹 프로그래밍은 특정 문제가 아래의 조건을 만족할 때 사용할 수 있다.
  1. **최적 부분 구조(Optimal Substructure)**
    - 큰 문제를 작은 문제로 나눌 수 있으며 작은 문제의 답을 모아서 큰 문제를 해결할 수 있는 구조이다.
  2. **중복되는 부분 문제(Overlapping Subproblem)**
    - 동일한 작은 문제를 반복적으로 해결해야 한다.
    - 즉 부분 문제가 중첩되어 여러 번 등장한다는 의미이다. 어떤 문제를 해결하기 위해서 동일한 작은 문제가 반복적으로 호출되어 동일한 문제를 반복적으로 해결해야 할 문제이다.
- 다이나믹 프로그래밍을 활용하여 해결할 수 있는 대표적인 문제가 피보나치 수열 문제이다.
  - 피보나치 수열은 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, … 와 같은 형태의 수열이다.
  - 특정 번째의 피보나치를 구하고자 할 때 앞에 있는 두 수를 더한 것이 해당 피보나치 수가 된다.
  - **점화식**이란 인접한 항들 사이의 관계식을 의미한다.
  - 피보나치 수열을 점화식으로 표현하면, $a_n = a_{n-1} + a_{n+2}, a_1 = 1, a_2 = 1$ 이다.
  - n 이 엄청 커도, 시작 부분 항에 대한 값을 알고 있으면 모든 항에 대해서 값을 구할 수 있다.
  - 이러한 점화식은 프로그래밍 상에서 재귀 함수를 이용해서 점화식을 그 형태 그대로 소스코드로 옮겨서 구현할 수 있다.
    
    ![Untitled](/assets/images/DS&Algorithm/Untitled%205.png){: .align-center}
    
  - 프로그래밍에서는 위 같은 수열을 배열이나 리스트를 이용해서 표현한다. 실제로 수열은 영어로 sequence 이며 일반적으로 선형적인 데이터를 표현할 때 사용할 수 있는 배열이나 리스트를 이용해서 수열에 대한 각각의 값들을 기록할 수 있다.
  - 따라서 수열에 대한 정보를 프로그램 상에서 메모리에 기록하고자 한다면 **배열이나 리스트**를 이용할 수 있다. 이러한 방법은 DP 에서도 그대로 사용된다.
  - DP 에서 각각의 계산된 결과를 담기 위해서 배열이나 리스트를 사용하고, 별도로 테이블과 같은 공간에 값을 기록한다고 하여 이러한 배열이나 리스트를 **테이블**이라고 부르기도 한다.
    ```python
    # 재귀함수로 피보나치 구현
    def fibo(x):
        # 종료 조건
        if x == 1 or x == 2:
            return 1
        return fibo(x-1) + fibo(x-2)
    ```
  - 재귀함수가 무한 루프를 돌지 않고 특정지점에서 멈출 수 있도록 구현하는 것이 일반적이다.
- 피보나치 수열의 시간 복잡도 분석
  - 점화식에 맞게 단순 재귀함수로 피보나치 수열을 해결하면 지수 시간 복잡도를 가지게 된다.
  - 지수 시간 복잡도는 n 의 값이 조금만 커지더라도 기하급수적으로 더 많은 수행 시간이 필요하게 된다. 따라서 효율성 테스트에서 쉽게 통과할 수 없다.
    
    ![Untitled](/assets/images/DS&Algorithm/Untitled%206.png){: .align-center}
    
  - 위 예시에서 보면 $f(2)$ 가 여러 번 호출되어 중복되는 부분 문제가 있는 걸 알 수 있다.
  - 따라서 한 번 해결한 문제에 대해서, 이미 해결한 적이 있기 때문에 반복되는 문제의 답을 알려줄 수 있도록 별도의 메모리 공간에 이미 해결한 문제에 대한 정보를 기록해두지 않으면, 중복되는 부분 문제를 계속해서 해결하게 되고 그렇기 때문에 수행시간 측면에서 매우 비효율적으로 동작하게 된다.
  - 재귀함수를 이용한 피보나치 수열의 시간 복잡도는 $O(2^N)$ 이다. $N$ 이 조금만 커져도 매우 높은 수행시간을 요구한다.
  - 따라서 DP 를 이용하여 시간 복잡도를 개선할 필요가 있다.
  - 피보나치 수열은 DP 의 사용 조건인 최적 부분 구조, 중복되는 부분 문제를 만족한다.
    - 큰 문제를 해결하기 위해서 작은 문제 두 개를 해결한 결과값을 가지고 있으면 된다.
    - 또 동일한 작은 문제가 반복적으로 해결될 수 있도록 요구된다.

#### 메모이제이션(Memoization)
- 메모이제이션은 다이나믹 프로그래밍을 구현하는 방법 중 하나이다.
- DP 는 상향식, 하향식으로 구현될 수 있는데, **메모이제이션은 Top down, 하향식에서 사용되는 방식**이다.
- 한 번 계산된 결과를 **별도의 메모리 공간에 메모**하는 기법을 의미한다.
- 같은 문제를 다시 호출하면 메모했던 결과를 그대로 가져오는 것이다.
- 별도의 테이블, 즉 배열에 값을 기록해 놓는다는 점에서 **캐싱(Caching)**이라고도 한다. 실제로 DP 를 이용해서 문제를 해결할 때 사용하는 배열 이름을 캐시 혹은 메모, 테이블, DP, D 라고 설정하기도 한다.

#### Top down vs. Bottom up
- Top down(메모이제이션) 방식은 하향식, bottom up 방식은 상향식이라고 한다.
  - Top down 방식은 구현 과정에서 **재귀함수**를 이용한다. 즉 큰 문제를 해결하기 위해서 작은 문제들을 재귀적으로 호출하여 그 작은 문제들이 모두 해결 되었을 때 실제로 큰 문제에 대한 답까지 얻을 수 있도록 코드를 작성한다. 그러한 과정에서 한 번 계산된 결과값을 기록하기 위해서 메모이제이션 기법을 이용한다.
  - Bottom up 방식은 아래부터 작은 문제를 하나씩 해결해가면서 먼저 계산한 문제들의 값을 활용해서 그 다음의 문제까지 차례대로 해결한다. 그래서 Bottom up 방식을 구현할 때는 프로그래밍 언어에서 **반복문**을 이용한다.
- DP 의 전형적인 형태는 bottom up 방식이다.
  - bottom up 방식에서 사용되는 결과 저장용 리스트는 DP 테이블이라고 부른다.
- 엄밀히 말하면 메모이제이션은 이전에 계산된 결과를 일시적으로 기록해 놓는 넓은 개념을 의미한다.
  - 따라서 메모이제이션은 DP 에 국한된 개념은 아니다.
  - 한 번 계산된 결과를 담아 놓기만 하고 DP 를 위해 활용하지 않을수도 있다.
  - 따라서 메모이제이션과 DP 는 엄밀히 말하면 서로 다른 개념이다. 즉 DP 를 구현하는 방법 중에서 하향식 방법으로 접근할 때 이미 계산된 결과를 기록하는 방법으로 메모이제이션을 활용하는 것이다.
- Top down DP 를 활용한 피보나치 수열 해결
    ```python
    # 한 번 계산된 결과를 메모이제이션(Memoization)하기 위한 리스트 초기화
    d = [0] * 100
    
    # 재귀함수로 피보나치 함수 구현(Top down DP)
    def fibo(x):
        # 종료 조건(1 혹은 2일 때 1을 반환)
        if x == 1 or x == 2:
            return 1
        
        # 이미 계산한 적 있는 문제라면 그대로 반환. DP table 의 값을 확인한다.
        if d[x] != 0:
            return d[x]
        
        # 아직 계산하지 않은 문제라면 점화식에 따라서 피보나치 결과 반환
        d[x] = fibo(x-1) + fibo(x-2)
        
        return d[x]
    ```
    - 위 코드처럼 별도의 리스트 혹은 배열을 만들어서 거기에 해당 문제가 해결된 적 있는 문제인지 아닌지, 그리고 해결되었다면 그 값을 기록할 수 있도록 하여 DP 를 구현할 수 있다.
- Bottom up DP 를 활용한 피보나치 수열 해결
    
    ```python
    # 앞서 계산된 결과를 저장하기 위한 DP 테이블 초기화
    d = [0] * 100
    
    # 첫번째 피보나치 수와 두번째 피보나치 수는 1
    d[1] = 1
    d[2] = 1
    n = 99
    
    # 반복문으로 피보나치 함수 구현 (Bottom up DP)
    for i in range(3, n+1):
        d[i] = d[i-1] + d[i-2]
    ```
    - 반복문을 이용해서 점화식의 값을 그대로 기입하여 차례대로 각각의 항에 대한 값을 구해나간다.
    - 즉 작은 문제부터 해결해놓고, 먼저 해결한 작은 문제들을 조합해서 앞으로의 큰 문제들을 차례대로 구해나가는 것이다.

#### 메모이제이션 동작 분석
- 피보나치 수열에서 이미 계산된 결과를 메모리에 저장하면 아래 그림과 같이 색칠된 노드만 처리할 것을 기대할 수 있다.

![Untitled](/assets/images/DS&Algorithm/Untitled%207.png){: .align-center}

- 굉장히 많은 수의 문제를 따로 해결할 필요가 있었는데 메모이제이션 기법을 통해서 실제 호출되는 함수들이 굉장히 적어진다.
- 메모이제이션 기법을 이용하는 경우 피보나치 수열 함수의 시간 복잡도는 $O(N)$ 이다.

![Untitled](/assets/images/DS&Algorithm/Untitled%208.png){: .align-center}

- 지수시간 만큼 걸리던 피보나치 문제를 DP 를 적용함으로써 선형시간까지 그 시간복잡도를 줄일 수 있다. 이미 계산된 결과는 바로 상수시간으로 리턴할 수 있도록 만들어지기 때문에 특정 번째의 피보나치 수를 구하는 과정에서 걸리는 총 시간은 상수시간($O(N)$) 이다.

#### DP vs. 분할 정복(divide and conquer)
- 퀵 정렬과 같은 분할 정복 아이디어와 비교했을 때 DP 는 어떤 차이점을 가질까?
- DP 와 분할 정복은 모두 **최적 부분 구조**를 가질 때 사용할 수 있다.
  - 큰 문제를 작은 문제로 나눌 수 있으며 작은 문제의 답을 모아서 큰 문제를 해결할 수 있는 상황에 사용할 수 있다.
- DP 와 분할 정복의 **차이점은 부분 문제의 중복**이다.
  - DP 에서는 각 부분 문제들이 서로 영향을 미치며 부분 문제가 중복된다.
    - 예를 들어 피보나치 수열 문제를 해결할 때 작은 문제가 반복적으로 호출되는 것을 확인할 수 있었다.
  - 분할 정복 문제에서는 동일한 부분 문제가 반복적으로 계산되지 않는다.
- 분할 정복의 대표적인 예시인 퀵 정렬을 살펴보자.
  - 한 번 기준 원소(Pivot)가 자리를 변경해서 자리를 잡으면 그 기준 원소의 위치는 바뀌지 않는다.
  - 분할 이후에 해당 피벗을 다시 처리하는 부분 문제는 호출하지 않는다.
    
    ![Untitled](/assets/images/DS&Algorithm/Untitled%209.png){: .align-center}
    
  - 실제로 퀵 정렬의 동작 원리를 확인해보면 분할이 이루어진 뒤에 왼쪽, 오른쪽 원소들에 대해 다시 한번 각각 재귀적으로 퀵 정렬을 수행하며, 재귀적으로 호출되는 과정이 모두 종료되었을 때 전체 범위에서 정렬이 수행된다.
  - 이처럼 한 번 분할이 이루어지고 나면 그 Pivot 은 다른 부분 문제에 포함되지 않고 그 위치가 더이상 변경되지 않기 때문에 부분 문제가 중복되지 않는다고 표현할 수 있다.
  - 이러한 특징 때문에 퀵 정렬은 분할정복으로 분류가 된다.
  - 퀵 정렬 뿐 아니라 병합 정렬 등 분할정복의 아이디어를 이용하는 다른 알고리즘들도 마찬가지이다.

#### DP 문제에 접근하는 방법
- 주어진 문제가 DP 유형임을 파악하는 것이 중요하다.
- 가장 먼저 그리디, 구현, 완전 탐색 등의 아이디어로 문제를 해결할 수 있는지 검토할 수 있다.
- 다른 알고리즘으로 풀이 방법이 떠오르지 않으면 DP 를 고려하는 것이다.
- 어떤 문제가 **작은 문제를 조합해서 큰 문제를 해결할 수 있는 형태**를 가지며, **부분 문제가 중복되는 특성**을 가진다고 하면 DP 를 적용하는 방식으로 문제에 접근할 수 있다.
- 일단 재귀 함수로 비효율적인 완전 탐색 프로그램을 작성한 뒤에 Top down 방식으로 작은 문제에서 구한 답이 큰 문제에서 그대로 사용될 수 있으면, 코드를 개선할 수 있다.
  - 메모이제이션 기법을 활용하여, 한 번 계산된 결과가 별도의 리스트 혹은 배열에 담기도록 하는 것이다.
- 일반적인 코딩 테스트 수준에서는 기본 유형의 DP 문제가 출제되는 경우가 많다.


#### 가장 긴 증가하는 부분 수열(Longest Increasing Subsequence, LIS)
- 가장 긴 증가하는 부분 수열(Longest Increasing Subsequence, LIS) 문제는 전형적인 DP 의 아이디어와 같다.
- array = {4, 2, 5, 8, 4, 11, 15} 가 있을 때 이 수열의 가장 긴 증가하는 부분 수열은 {4, 5, 8, 11, 15} 이다.
- $D[i]$ 가 $array[i]$ 를 마지막 원소로 가지는 부분 수열의 최대 길이라고 했을 때, 점화식은 모든 $0 ≤ j < i$ 에 대하여,
$D[i]$ = $max(D[i], D[j]+1) \; \;  if \; \; array[j] < array[i]$ 로 표현할 수 있다.
- 마지막 원소로 가지는 부분 수열의 최대 길이를 구하기 위해서 앞에 있는 원소($j$)들을 매번 확인해야 하기 때문에 시간복잡도는 최악의 경우 $O(N^2)$ 이 될 수 있다.

![Untitled](/assets/images/DS&Algorithm/Untitled%2010.png){: .align-center}

- 각 원소별로 점화식에 따라 DP 테이블을 갱신시켜준다.
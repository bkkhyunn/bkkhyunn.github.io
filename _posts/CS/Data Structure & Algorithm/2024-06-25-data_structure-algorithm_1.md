---
title: "[Algorithm] 1. Greedy Algorithm"
excerpt: "그리디 알고리즘에 대해 알아보자"

categories: "data_structure-algorithm"
tags:
    - algorithm
    - greedy
toc: true  
toc_sticky: true
toc_label: "Contents In Page"
author_profile: true
use_math: true

date: 2024-06-25
---

## 그리디 알고리즘(Greedy Algorithm)
- 탐욕법이라고도 부른다. **현재 상황에서 지금 당장 좋은 것만 고르는 방법**이다.
- 그리디 알고리즘은 일반적으로 문제를 풀기 위한 최소한의 아이디어를 떠올릴 수 있는 능력을 요구한다.
- 그리디 알고리즘은 그 **정당성** 분석이 중요하다. 단순히 가장 좋아 보이는 것을 반복적으로 선택해도 최적의 해를 구할 수 있는지를 검토해야 한다. 왜냐하면 일반적 상황에서 그리디 알고리즘은 최적의 해를 보장할 수 없을 때가 많기 때문이다.
- 그러나 코딩테스트 대부분은 탐욕법으로 얻은 해가 최적의 해가 되는 상황에서 이를 추론할 수 있어야 풀리도록 출제된다.

## 예제
#### 거스름돈 문제
- 거스름돈을 가장 적은 동전의 개수로 거슬러주는 문제가 대표적인 그리디 알고리즘 문제이다.
- 이 문제가 그리디 알고리즘으로 최적의 해를 보장하는 정당성이 무엇일까?
- 500원, 100원, 50원, 10원 등 가지고 있는 동전들은 큰 단위가 항상 작은 단위의 배수이므로 작은 단위의 동전들을 종합해서 더 적은 동전의 개수를 가진 해가 나올 수 없기 때문이다.
- 예를 들어 화폐단위가 500원, 400원, 100원 이라면 그리디 알고리즘으로 최적의 해를 보장할 수 없다. 큰 단위가 작은 단위의 배수가 아니기 때문이다.
    <br>
    ```python
    n = 1260
    count = 0

    # 큰 단위의 화폐부터 차례대로 확인하기
    coin_types = [500, 100, 50, 10]

    for coin in coin_types:
        count += n // coin # 해당 화폐로 거슬러 줄 수 있는 동전의 개수 세기
        n %= coin

    print(count)
    ```
- 이처럼 **그리디 알고리즘 문제에서는 문제 풀이를 위한 최소한의 아이디어를 떠올리고 이것이 정당한지 검토할 수 있어야 한다.**
- 위 거스름돈 예제에서 그리디 알고리즘이 최적의 해를 보장할 수 없다면, DP 를 사용할 수 있다.

#### Fractional Knapsack Problem
- 물건을 쪼갤 수 있는 배낭 문제이다.
- 가치가 높은 순으로 정렬한 뒤 배낭에 담고, 남은 부분은 물건을 쪼개어 넣어 조합을 찾는다.
- 대표적인 그리디 알고리즘으로 해결할 수 있는 문제이다.
    <br>
    ```python
    # 이익, 무게
    cargo = [
        (4, 12),
        (2, 1),
        (10, 4),
        (1, 1),
        (2, 2),
    ]


    def fractional_knapsack(cargo):
        capacity = 15
        pack = []
        # 단가(이익 / 무게) 계산 역순 정렬
        for c in cargo:
            pack.append((c[0] / c[1], c[0], c[1]))
        pack.sort(reverse=True)

        # 단가 순 그리디계산
        total_value: float = 0
        for p in pack:
            if capacity - p[2] >= 0:
                capacity -= p[2]
                total_value += p[1]
            # 가방의 용량을 넘어서는 경우, 가방의 남은 용량만큼만 짐을 넣는다.
            else:
                fraction = capacity / p[2]
                total_value += p[1] * fraction
                break

        return total_value


    r = fractional_knapsack(cargo)
    print(r)
    ```
- 만일 물건을 쪼갤 수 없는 Knapsack problem 은 DP 를 사용해서 해결 가능하다.
#### 주유소 문제
- 주유 비용을 최소로 하는 문제이다.
- 현재 도시에서 주유할 수 있는 비용과 다음 도시에서 주유할 수 있는 비용을 비교해서 적은 비용을 유지하면서 거리를 곱한다.
    <br>
    ```python
    import sys
    from collections import deque

    n = int(input())
    dist = deque(list(map(int,sys.stdin.readline().split())))
    cost = deque(list(map(int,sys.stdin.readline().split())))

    # 맨 처음 도시에서는 비용에 관계없이 주유해야 한다.
    now_cost = cost.popleft()
    now_dist = dist.popleft()
    total_cost = now_cost * now_dist

    while dist:
        next_cost = cost.popleft()
        if now_cost > next_cost:
            now_cost = next_cost
            
        now_dist = dist.popleft()
        total_cost += now_dist * now_cost

    print(total_cost)
    ```
- 이 문제 또한 매번 적은 비용을 선택해서 총 주유량을 구하기 때문에 그리디에 속한다.

#### 그 외
- 그리디 알고리즘에 속하는 대표적 알고리즘
  - 최소 신장 트리: 크루스칼 알고리즘, 프림 알고리즘
  - 최소경로 알고리즘: 다익스트라 알고리즘
  - 문서 압축 알고리즘: 허프만 코딩 알고리즘
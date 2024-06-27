---
title: "[Algorithm] 9. Graphs - Union Find"
excerpt: "서로소 집합에 대해 알아보자"

categories: "data_structure-algorithm"
tags:
    - algorithm
    - graphs
    - Disjoint Sets
    - Union Find
toc: true  
toc_sticky: true
toc_label: "Contents In Page"
author_profile: true
use_math: true

date: 2024-06-27
---

### 서로소 집합(Disjoint Sets)
- 서로소 집합(Disjoint Sets) 이란 공통 원소가 없는 두 집합을 의미한다.

![Untitled](/assets/images/DS&Algorithm/Untitled%2024.png){: .align-center}

- 흔히 두 집합이 서로소 관계인지를 판단할 때 원소를 확인해서 판단을 진행할 수 있다.

#### 서로소 집합 자료구조
- 서로소 판별을 위해 사용할 수 있는 자료구조다. 서로소 부분 집합들로 나누어진 원소들의 데이터를 처리하기 위한 자료구조로 사용할 수 있다.
- 서로소 집합 자료구조는 두 종류의 연산을 지원한다.
  - **합집합(Union)**: 두 개의 원소가 포함된 집합을 하나의 집합으로 합치는 연산이다. 즉 합집합 연산을 수행할 때는 두 원소가 포함되고 있는 각각의 집합이 같은 집합이 될 수 있도록 합친다.
  - **찾기(Find)**: 특정한 원소가 속한 집합이 어떤 집합인지 알려주는 연산이다. 예를 들어 두 원소가 주어졌을 때 두 원소가 같은 집합에 포함되어 있는지에 대한 여부를 이 찾기 연산으로 알 수 있다.
- 서로소 집합 자료구조는 **합치기 찾기(Union Find) 자료구조**라고 불리기도 한다.
- 여러 개의 합치기 연산이 주어졌을 때 서로소 집합 자료구조의 동작 과정은 다음과 같다.
    1. 합집합(Union) 연산을 확인하여 서로 연결된 두 노드 A, B 를 확인한다.
        
        1) A 와 B 의 루트 노드 A’, B’ 를 각각 찾는다.
        
        2) A’ 를 B’ 의 부모 노드로 설정한다.
        
    2. 모든 합집합(Union) 연산을 처리할 때까지 1번의 과정을 반복한다.
- 자세하게 살펴보면, 먼저 노드의 개수 크기의 부모 테이블을 초기화 한다. 이 때 맨 처음 부모는 자기 자신으로 둔다. 즉 6개 모두 서로 각각 집합이라고 보는 것이다.

    ![Untitled](/assets/images/DS&Algorithm/Untitled%2025.png){: .align-center}

- 일반적으로 합치기 연산을 수행할 때는 더 큰 루트노드가 더 작은 루트노드를 가리키도록 만들어서 테이블을 갱신하는 것이 관행처럼 사용된다.

    ![Untitled](/assets/images/DS&Algorithm/Untitled%2026.png){: .align-center}

- 실제로 두 원소가 같은 집합에 포함되어 있는지 확인하기 위해서 최종 루트 노드를 찾아야 한다.

    ![Untitled](/assets/images/DS&Algorithm/Untitled%2027.png){: .align-center}

- 즉 위에서 합치기 연산을 다 수행하고 나서, 3번 노드와 4번 노드를 볼 때 부모 노드는 다르지만, 최종 루트노드가 1이기 때문에 같은 집합에 속한다는 것을 알 수 있다.
- 이렇게 연결을 다 끝내고 나면, 서로소 집합 자료구조에서는 연결성을 통해 손쉽게 집합의 형태를 확인할 수 있다.
- 하나의 루트 노드를 가지고 하나의 트리 형태로 구성된 그래프는 같은 집합이라고 표현할 수 있다.
- 다만 위처럼 **기본적인 형태의 서로소 집합 자료구조에서는 루트 노드에 즉시 접근할 수 없다.** 왜냐하면 루트 노드를 찾기 위해 부모 테이블을 계속해서 확인해가며 거슬러 올라가야 하기 때문이다. 위에서는 3번 노드의 루트 노드는 1이지만 부모 노드 2를 거쳐 루트 노드 1을 확인할 수 있다.
- 즉 부모 테이블 자체에는 부모 노드만 설정되어 있기 때문에 실제 함수 상에서 재귀적으로 거슬러 올라가며 루트 노드가 무엇인지 확인해야 한다.

    ```python
    # 특정 원소가 속한 집합을 찾기 -> 루트 노드를 반환한다.
    def find_parent(parent, x):
        # 루트 노드를 찾을 때까지 재귀 호출
        if parent[x] != x:
            return find_parent(parent, parent[x])
        return x
        
    # 두 원소가 속한 집합을 합치기
    def union_parent(parent, a, b):
        a = find_parent(parent, a)
        b = find_parent(parent, b)
        if a < b:
            parent[b] = a
        else:
            parent[a] = b
            
    # 노드의 개수 v, 간선(Union 연산)의 개수 e

    # 부모 테이블 초기화하기
    parent = [0] * (v+1)

    # 부모 테이블 상에서, 부모를 자기 자신으로 초기화
    for i in range(1, v+1):
        parent[i] = i
        
    # Union 연산을 각각 수행
    for i in range(e):
        a, b = map(int, input().split())
        union_parent(parent, a, b)
        
    # 각 원소가 속한 집합 출력 -> 그 집합의 루트 노드를 출력한다. 루트가 같으면 같은 집합이다.
    for i in range(1, v+1):
        print(find_parent(parent, i), end=' ')

    # 부모 테이블 내용 출력 -> 자신의 루트, 집합에 대한 정보가 아닐 수 있다.
    for i in range(1, v+1):
        print(parent[i], end=' ')
    ```

#### 기본적인 형태의 서로소 집합 자료구조 구현 방법의 문제점
- 수행 시간 측면에서 문제점이 있다.
- 합집합(Union) 연산이 편향되게 이루어지는 경우 찾기(Find) 함수가 비효율적으로 동작한다.
- 찾기(Find) 함수는 매번 현재 노드에서 부모 테이블을 참조하여 그 부모가 자기 자신이 아니라면 루트가 아니라는 의미이기 때문에 자신의 부모에 대해서 다시 한번 재귀적으로 호출해서 확인하는 과정을 반복한다.
- 최악의 경우에는 찾기(Find) 함수가 모든 노드를 다 확인하게 되어 시간 복잡도가 $O(V)$ 이다.
- 예시로, {1, 2, 3, 4, 5} 의 총 5개의 원소가 존재하는 상황을 확인해보자. 모든 원소가 같은 집합에 속한다.

    ![Untitled](/assets/images/DS&Algorithm/Untitled%2028.png){: .align-center}

- 5번 노드에 대한 루트 노드가 무엇인지 알기 위해서 Find 함수를 호출하게 되면 모든 노드를 다 확인하게 된다.
- 이처럼 단순히 부모 테이블을 확인하면서 거슬러 올라가는 방식을 이용하면 수행시간 측면에서 매우 비효율적으로 동작한다.

#### 경로 압축
- 따라서 이러한 찾기(Find) 함수를 최적화하기 위한 방법으로 **경로 압축(Path Compression)**을 이용할 수 있다.
- 즉 루트 노드까지 도달하기 위한 경로를 압축해서 시간을 단축한다는 의미이다.
- 찾기(Find) 함수를 재귀적으로 호출한 뒤에 부모 테이블 값을 바로 갱신한다.

    ```python
    # 특정 원소가 속한 집합을 찾기
    def find_parent(parent, x):
        # 루트 노드가 아니라면, 루트 노드를 찾을 때까지 재귀적으로 호출
        if parent[x] != x:
            # 부모 테이블 값을 루트 노드로 바로 갱신
            parent[x] = find_parent(parent, parent[x])
        return parent[x]
    ```

- 결과적으로 `find_parent` 함수를 호출한 뒤에 부모 테이블에 적혀있는 부모의 값이 자신의 루트가 될 수 있도록 설정하는 것이다.
- 다시 말해, Find 연산을 수행한 뒤에는 부모 테이블의 값이 다시 루트가 될 수 있도록 하여 Find 함수를 호출할 때 사용했던 노드에 대한 부모의 값이 루트와 동일하도록 경로가 압축되는 것이다.
- 경로 압축 기법을 적용하면 각 노드에 대해서 찾기(Find) 함수를 호출한 이후에 해당 노드의 루트 노드가 바로 부모 노드가 된다.
- 동일한 예시에 대해서 모든 합집합(Union) 함수를 처리한 후 각 원소에 대하여 찾기(Find) 함수를 수행하면 부모 테이블이 갱신된다.

    ![Untitled](/assets/images/DS&Algorithm/Untitled%2029.png){: .align-center}

- 기본적인 방법에 비하여 시간 복잡도가 개선된다.

    ```python
    # 특정 원소가 속한 집합을 찾기
    def find_parent(parent, x):
        # 루트 노드가 아니라면, 루트 노드를 찾을 때까지 재귀적으로 호출
        if parent[x] != x:
            # 부모 테이블 값을 루트 노드를 갖도록 바로 갱신
            parent[x] = find_parent(parent, parent[x])
        return parent[x]
        
    # 두 원소가 속한 집합을 합치기
    def union_parent(parent, a, b):
        a = find_parent(parent, a)
        b = find_parent(parent, b)
        if a < b:
            parent[b] = a
        else:
            parent[a] = b
            
    # 노드의 개수 v, 간선(Union 연산)의 개수 e

    # 부모 테이블 초기화하기
    parent = [0] * (v+1)

    # 부모 테이블 상에서, 부모를 자기 자신으로 초기화
    for i in range(1, v+1):
        parent[i] = i
        
    # Union 연산을 각각 수행
    for i in range(e):
        a, b = map(int, input().split())
        union_parent(parent, a, b)
        
    # 각 원소가 속한 집합 출력 -> 그 집합의 루트 노드를 출력한다. 루트가 같으면 같은 집합이다.
    for i in range(1, v+1):
        print(find_parent(parent, i), end=' ')

    # 부모 테이블 내용 출력 -> 자신의 루트, 집합에 대한 정보가 아닐 수 있다.
    for i in range(1, v+1):
        print(parent[i], end=' ')
    ```

#### 서로소 집합을 활용한 사이클 판별 알고리즘
- 서로소 집합은 **무방향 그래프 내에서의 사이클을 판별**할 때 사용할 수 있다.
- 다시 말해 방향성이 없는 그래프에서는 서로소 집합 자료구조를 사이클 판별 목적으로 사용할 수 있다.
- 참고로 방향 그래프에서의 사이클 여부는 DFS 를 이용해서 판별할 수 있다.
- 무방향 그래프에서의 사이클 판별 알고리즘은,
    1. 각 간선을 하나씩 확인하며 두 노드의 루트 노드를 확인한다. 즉 Find 함수를 호출한다.
        
        1) 루트 노드가 서로 다르다면, 즉 서로 다른 집합이라면 두 노드에 대하여 합집합(Union) 연산을 수행하여 같은 집합이 될 수 있도록 한다.
        
        2) 루트 노드가 서로 같다면, 즉 이미 같은 집합이라면 사이클(Cycle)이 발생한 것이다.
        
    2. 그래프에 포함되어 있는 모든 간선에 대하여 1번 과정을 반복한다.
- 모든 간선이 다 처리될 때까지 사이클이 발생하지 않았다면 해당 그래프에는 사이클이 없다고 판단할 수 있다.
- 모든 노드에 대하여 자기 자신을 부모로 설정하는 형태로 부모 테이블을 초기화한다.

    ![Untitled](/assets/images/DS&Algorithm/Untitled%2030.png){: .align-center}

- 간선 (1, 2)를 확인한다. 노드 1과 노드 2의 루트 노드는 각각 1과 2이다. 따라서 더 큰 번호에 해당하는 노드 2의 부모 노드를 1로 변경한다.

    ![Untitled](/assets/images/DS&Algorithm/Untitled%2031.png){: .align-center}

- 즉 노드 1과 노드 2에 대해서 같은 집합에 속하도록 처리하는 것이다.
- 다음 간선인 (1, 3) 을 확인한다. 마찬가지로 1번 노드와 3번 노드에 각각 Find 함수를 호출한 뒤에 그렇게 찾은 루트 노드에 대해서 합치기(Union) 연산을 수행해서 같은 집합에 속하도록 만들어준다.
- 마지막 간선(2, 3)을 확인한다. 이미 노드 2와 노드 3의 루트 노드는 모두 1이다. 다시 말해 이미 같은 집합에 속한다는 것을 알 수 있다. 즉 이 간선을 확인했을 때 현재 그래프에 사이클이 발생했다는 것을 알 수 있다.
- 이렇게 간선을 하나씩 확인하면서 합치기 연산을 수행하는 것 만으로도 사이클을 판별할 수 있다는 것을 알 수 있다.

    ```python
    # 특정 원소가 속한 집합을 찾기 (경로 압축)
    def find_parent(parent, x):
        # 루트 노드가 아니라면, 루트 노드를 찾을 때까지 재귀적으로 호출
        if parent[x] != x:
            # 부모 테이블 값을 루트 노드를 갖도록 바로 갱신
            parent[x] = find_parent(parent, parent[x])
        return parent[x]
        
    # 두 원소가 속한 집합을 합치기
    def union_parent(parent, a, b):
        a = find_parent(parent, a)
        b = find_parent(parent, b)
        if a < b:
            parent[b] = a
        else:
            parent[a] = b
            
    # 노드의 개수 v, 간선(Union 연산)의 개수 e

    # 부모 테이블 초기화하기
    parent = [0] * (v+1)

    # 부모 테이블 상에서, 부모를 자기 자신으로 초기화
    for i in range(1, v+1):
        parent[i] = i
        
    # Cycle 발생 여부
    cycle = False

    for i in range(e):
        a, b = map(int(), input().split())
        
        # 사이클이 발생한 경우 종료
        if find_parent(parent, a) == find_parent(parent, b):
            cycle = True
            break
        # 사이클이 발생하지 않았다면 합집합(Union) 연산 수행
        # 즉 노드 a 와 노드 b 가 같은 집합에 속해있지 않다면 같은 집합에 속하도록 합집합 연산 수행
        else:
            union_parent(parent, a, b)

    if cycle:
        print("사이클 발생")
    else:
        print("사이클 발생하지 않음")
    ```
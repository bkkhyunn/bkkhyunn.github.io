---
title: "[Algorithm] 10. Graphs - Minimum Spanning Tree(MST)"
excerpt: "최소 신장 트리에 대해 알아보자"

categories: "data_structure-algorithm"
tags:
    - algorithm
    - graphs
    - MST
    - Kruskal
    - Prim
toc: true  
toc_sticky: true
toc_label: "Contents In Page"
author_profile: true
use_math: true

date: 2024-06-27 13:55
---

### 최소 신장 트리 알고리즘(MST)
- 신장 트리란 그래프에서 모든 노드를 포함하면서 사이클이 존재하지 않는 부분 그래프를 의미한다.
    
    ![Untitled](/assets/images/DS&Algorithm/Untitled%2032.png){: .align-center}
    
- 즉 원본 그래프에 존재하는 간선을 모두 활용하지는 않고 일부의 간선만 활용해서 모든 노드가 포함되어 있는 하나의 부분 그래프를 만드는 것이다.
- **모든 노드가 포함되어 서로 연결되면서 사이클이 존재하지 않는다는 조건은 트리의 조건**이기도 하다. 그래서 신장 트리라고 부른다.
- 신장 트리는 모든 노드가 다 연결은 되어 있지만 일부 간선을 사용하지 않아도 괜찮도록 해준다는 점에서 실제 문제 상황에서도 이러한 아이디어가 효과적으로 사용될 수 있는 경우가 존재한다.
- 최소한의 비용으로 구성되는 신장트리를 찾을 때는 어떻게 해야 할까? 이 문제가 바로 **최소 신장 트리 문제**다.
  - 예를 들어 N개의 도시가 존재하는 상황에서 두 도시 사이에 도로를 놓아 전체 도시가 서로 연결될 수 있게 도로를 설치하는 경우를 생각해보자.
  - 두 도시 A, B 를 선택했을 때 A 에서 B 로 이동하는 경로가 반드시 하나 이상은 존재하도록 도로를 설치한다.
  - 즉 모든 노드가 서로 연결되어 이동이 가능하도록 만들되 최소한의 비용으로 전체 신장 트리를 구성하고자 하는 문제가 최소 신장 트리 문제이다.

#### 크루스칼(Kruscal) 알고리즘
- 대표적인 최소 신장 트리 알고리즘이다.
- 그리디 알고리즘으로 분류된다.
- 동작 과정
    1. 간선 데이터를 비용에 따라 오름차순으로 정렬한다.
    2. 간선을 하나씩 확인하며 현재의 간선이 사이클을 발생시키는지 확인한다.

        1) 사이클이 발생하지 않는 경우 최소 신장 트리에 포함시킨다.
        
        2) 사이클이 발생하는 경우 최소 신장 트리에 포함시키지 않는다.

    3. 모든 간선에 대하여 2번 과정을 반복한다.
- 그래프의 모든 간선 정보에 대하여 비용을 기준으로 오름차순 정렬을 수행한다.

    ![Untitled](/assets/images/DS&Algorithm/Untitled%2033.png){: .align-center}

- 총 간선의 개수는 9개이다. 최종적으로 만들어지는 최소 신장 트리에 포함되어 있는 간선의 개수는 전체 노드의 개수 -1 이다. 이는 기본적으로 트리가 가지는 특성이며 사이클 또한 존재하지 않는다는 특징이 있다.
- 아직 처리하지 않은 간선중에 가장 짧은 간선을 선택하여 처리한다. 이 때 3번 노드와 4번 노드는 현재 같은 집합에 속해있지 않은 상태였기 때문에 서로 같은 집합에 속할 수 있도록 Union 함수를 호출한다.
- 그 다음 처리하지 않은 간선 중 비용이 적은 간선을 확인한다. 마찬가지로 같은 집합에 속해 있지 않다면 Union 연산을 수행한다.
- 다음에 확인하는 비용이 적은 간선의 두 노드가 이미 같은 집합에 속해 있다면, 이 간선을 포함시킨다면 사이클이 발생한다. 따라서 해당 간선은 무시하고 넘어간다.
- 이 과정을 반복한다.

    ![Untitled](/assets/images/DS&Algorithm/Untitled%2034.png){: .align-center}

- 결과적으로 알고리즘을 수행한 결과, 최소 신장 트리에 포함되어 있는 간선의 비용만 모두 더하면 그 값이 최종 비용에 해당한다.

    ```python
    # 크루스칼 알고리즘
    # 기본적으로 두 원소가 서로 같은 집합인지 아닌지를 판단하기 위해서 서로소 집합 자료구조 사용
    # 특정 원소가 속한 집합을 찾기
    def find_parent(parent, x):
        # 루트 노드를 찾을 때까지 재귀 호출. 경로 압축 사용
        if parent[x] != x:
            parent[x] = find_parent(parent, parent[x])
        return parent[x]
        
    # 두 원소가 속한 집합 합치기
    def union_parent(parent, a, b):
        a = find_parent(parent, a)
        b = find_parent(parent, b)
        if a < b:
            parent[b] = a
        else:
            parent[a] = b
            
    # 노드의 개수 v, 간선의 개수 e

    # 부모 테이블 초기화
    parent = [0] * (v+1)

    # 모든 간선을 담을 리스트와 최종 비용을 담을 변수
    edges, result = [], 0

    # 부모 테이블 상에서, 부모를 자기 자신으로 초기화
    for i in range(1, v+1):
        parent[i] = i

    # 모든 간선에 대한 정보를 입력 받기
    for _ in range(e):
        a, b, cost = map(int, input().split())
        # 비용 순으로 정렬하기 위해서 튜플의 첫번째 원소를 비용으로 설정
        edges.append((cost, a, b))
        
    # 간선을 비용 순으로 정렬
    edges.sort()

    # 간선을 하나씩 확인하며
    for edge in edges:
        cost, a, b = edge
        # 사이클이 발생하지 않는 경우에만 집합에 포함
        if find_parent(parent, a) != find_parent(parent, b):
            union_parent(parent, a, b)
            result += cost

    print(result)
    ```

##### 크루스칼 알고리즘 성능 분석
- 크루스칼 알고리즘은 간선의 개수가 $E$ 일 때, $O(ElogE)$의 시간 복잡도를 가진다.
- 그 이유는 크루스칼 알고리즘에서 가장 많은 시간을 요구하는 곳은 **간선의 정렬**을 수행하는 부분이기 때문이다.
- 표준 라이브러리를 이용해 $E$ 개의 데이터를 정렬하기 위한 시간 복잡도는 $O(ElogE)$ 이다.

#### 프림(Prim) 알고리즘
- 최소 신장 트리 알고리즘 중 하나이다.
- 프림 알고리즘도 크루스칼 알고리즘과 같이 그리디 알고리즘을 기반으로 하지만, 간선을 중심으로 최소 신장 트리를 구하는 크루스칼 알고리즘과 달리 **노드를 중심**으로 최소 신장 트리를 구한다.
- 임의의 노드를 선택하여 그래프를 만들고 해당 그래프와 연결되어 있는 노드 중 가장 작은 가중치의 간선으로 이어진 노드를 선택하여 그래프에 포함시킨다. 이미 그래프에 포함된 노드는 제외한다.
- 그래프에 속하지 않은 연결된 노드 중 간선의 가중치가 작은 순으로 노드를 그래프에 포함시켜야 하기 때문에 그래프와 연결된 노드들을 간선의 가중치를 우선순위로 하는 우선순위 큐를 가지고 프림 알고리즘을 적용시킨다.
- 이 때 그래프에 속한 노드인지 아닌지를 테이블을 하나 마련하여 확인한다.
- 동작과정
    1. 시작 노드를 우선순위 큐에 삽입한다. 시작 노드는 간선 가중치가 0이 된다.
    2. 우선순위 큐에서 노드를 하나 빼서 그래프에 포함시킨다. 해당 노드와 연결되어 있으면서 그래프에 포함되지 않은 노드들을 우선순위 큐에 삽입한다.
    3. 우선순위 큐에서 노드를 뺐을 때 그래프에 속해 있다면 아무 작업도 하지 않는다.
    4. 우선순위 큐가 비게 되면 종료한다.

    ```python
    # 크루스칼 알고리즘은 간선 중심, 프림 알고리즘은 정점 중심으로 동작한다.

    import heapq

    def prim(v, adj):
        # 그래프 (방문 처리와 같음)
        visited = [False] * (v + 1)
        min_heap = [(0, 1)]  # (cost, start_node)
        result = 0
        edges_used = 0 # 사용된 간선 개수(간선 개수는 항상 노드 개수보다 1만큼 작다)
        
        while min_heap and edges_used < v:
            cost, node = heapq.heappop(min_heap)
            if visited[node]:
                continue
            visited[node] = True
            result += cost
            edges_used += 1
            
            for next_cost, next_node in adj[node]:
                if not visited[next_node]:
                    heapq.heappush(min_heap, (next_cost, next_node))
                    
        return result

    v = 4  # 노드의 개수
    adj = {
        1: [(1, 2), (3, 3)],
        2: [(1, 1), (3, 3), (6, 4)],
        3: [(3, 1), (3, 2), (2, 4)],
        4: [(6, 2), (2, 3)]
    }

    print("Prim's MST cost:", prim(v, adj))
    ```

##### 프림 알고리즘 성능 분석
- 인접 행렬로 구성된 그래프에서 모든 간선에 대해 우선순위 큐 삽입 연산을 진행해야 한다.
- 따라서 $O(ElogV)$ 이다.
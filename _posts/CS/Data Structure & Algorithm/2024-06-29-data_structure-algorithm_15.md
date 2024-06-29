---
title: "[Algorithm] 15. Lowest Common Ancestor(LCA)"
excerpt: "최소 공통 조상 알고리즘에 대해 알아보자"

categories: "data_structure-algorithm"
tags:
    - algorithm
    - Tree
    - LCA
toc: true  
toc_sticky: true
toc_label: "Contents In Page"
author_profile: true
use_math: true

date: 2024-06-29 14:00
---

### 최소 공통 조상(Lowest Common Ancestor, LCA) 알고리즘
- 문제 : N(2≤N≤50,000) 개의 정점으로 이루어진 트리가 주어진다. 트리의 각 정점은 1번부터 N 번까지 번호가 매겨져 있으며, 루트는 1번이다. 두 노드의 쌍 M(1≤M≤10,000)개가 주어졌을 때, 두 노드의 가장 가까운 공통 조상이 몇 번인지 출력한다.
  - 이 문제를 최소 공통 조상 문제라고 할 수 있다.
  - 이 문제는 시간 복잡도 $O(NM)$ 으로 설계하면 문제를 통과할 수 있다.
- 최소 공통 조상(LCA) 문제는 두 노드의 공통된 조상 중에서 가장 가까운 조상을 찾는 문제이다.
- **최소 공통 조상은 가장 낮은 조상**이라고 볼 수 있다. 즉 **두 노드에서 거슬러 올라가면서 나타나는 조상 중 가장 낮은 위치에 존재하는 공통 조상**이다.
- 최소 공통 조상 알고리즘 동작과정
  1. 모든 노드에 대한 깊이(Depth)를 계산한다.
  2. 최소 공통 조상을 찾을 두 노드를 확인한다.
    1. 먼저 두 노드의 깊이(Depth)가 동일하도록 거슬러 올라간다.
    2. 이후에 부모가 같아질 때까지 반복적으로 두 노드의 부모 방향으로 거슬러 올라간다.
  3. 모든 LCA(a, b) 연산에 대하여 2번의 과정을 반복한다.
- DFS 를 이용해서 모든 노드에 대해서 깊이(depth)를 계산할 수 있다.
- 가장 먼저 두 노드의 깊이를 맞춘다. 이후에 두 노드에 대해서 동시에 거슬러 올라간다.
- 결과적으로 맨 처음 두 노드가 만나는 곳이 최소 공통 조상이다.

```python
import sys
sys.setrecursionlimit(int(1e5))
n = int(input())

parent = [0] * (n+1) # 부모 노드 정보
d = [0] * (n+1) # 각 노드까지의 깊이
c = [0] * (n+1) # 각 노드의 깊이가 계산되었는지 여부
graph = [[] for _ in range(n+1)] # 그래프 정보

for _ in range(n-1):
    a, b = map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)

# 루트노드부터 시작해서 깊이를 구하는 함수
def dfs(x, depth):
    c[x] = True
    d[x] = depth
    for y in graph[x]:
        if c[y]: # 이미 깊이를 구했다면 넘어가기
            continue
        parent[y] = x
        
        dfs(y, depth + 1)
        
# A 와 B 의 최소 공통 조상을 찾는 함수
def lca(a, b):

    # 먼저 깊이가 동일하도록 한다. 깊이가 더 큰 쪽을 올린다.
    while d[a] != d[b]:
        if d[a] > d[b]:
            a = parent[a]
        else:
            b = parent[b]
    
    # 노드가 같아지도록 거슬러 올라간다.
    while a != b:
        a = parent[a]
        b = parent[b]
    
    return a
    
dfs(1, 0) # 루트 노드는 1번 노드

m = int(input())

for i in range(m):
    a, b = map(int, input().split())
    print(lca(a, b))
```

#### 기본적인 LCA 알고리즘의 시간 복잡도 분석
- 매 쿼리마다 부모 방향으로 거슬러 올라가기 위해서 최악의 경우 $O(N)$ 의 시간복잡도가 요구된다.
- 따라서 모든 쿼리를 처리할 때의 시간 복잡도는 $O(NM)$ 이다.

#### 개선된 LCA 알고리즘
- LCA 심화 문제
  - N(2≤ N≤100,000) 개의 정점으로 이루어진 트리가 주어진다. 트리의 각 정점은 1번부터 N 번까지 번호가 매겨져 있고, 루트는 1번이다. 두 노드의 쌍 M(1≤M≤100,000)개가 주어졌을 때, 두 노드의 가장 가까운 공통 조상이 몇 번인지 출력한다.
  - 위 문제에서 동일한 로직으로 시간초과 판정을 받게 된 문제이다.
  - 따라서 시간을 단축시켜야 한다.
- 각 노드가 거슬러 올라가는 속도를 빠르게 만드는 방법을 고민해보자
  - 2의 제곱 형태로 거슬러 올라가도록 하면 $O(logN)$ 의 시간복잡도를 보장할 수 있다.
  - 즉, 메모리를 조금 더 사용하여 각 노드에 대하여 $2^i$ 번째 부모에 대한 정보를 기록한다.
  - 이를 위해 모든 노드에 대하여 깊이와 $2^i$ 번째 부모에 대한 정보를 계산한다.
  - DFS 를 호출해서 각각 자신의 한단계 위쪽 부모를 확인한다. 그 정보를 이용해서 DP 를 이용해 각각의 $2^i$ 번째 부모에 대한 정보를 구할 수 있다.
  - 먼저 두 노드의 깊이를 맞춘다.
  - 이후에 거슬러 올라간다. 이 때 2의 제곱 꼴로 빠르게 올라간다.
    
```python
import sys
input = sys.stdin.readline
sys.setrecursionlimit(int(1e5))
LOG = 21 # 2^20 = 1,000,000

n = int(input())
parent = [[0] * LOG for _ in range(n+1)] # 부모 노드 정보
d = [0] * (n+1) # 각 노드까지의 깊이
c = [0] * (n+1) # 각 노드의 깊이가 계산되었는지 여부
graph = [[] for _ in range(n+1)] # 그래프 정보

for _ in range(n-1):
    a, b = map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)

# 루트노드부터 시작해서 깊이를 구하는 함수
def dfs(x, depth):
    c[x] = True
    d[x] = depth
    for y in graph[x]:
        if c[y]: # 이미 깊이를 구했다면 넘어가기
            continue
        
        # 2^0. 한칸 위에 있는 부모 정보만 먼저 구한다.
        parent[y][0] = x
        
        dfs(y, depth + 1)
        
# 전체 부모 관계를 설정하는 함수
# 여기서 DP 를 이용해 2의 제곱꼴로 건너 뛰었을 때의 부모값들을 기록한다.
def set_parent():
    dfs(1, 0) # 루트 노드는 1번 노드
    for i in range(1, LOG):
        for j in range(1, n+1):
            # j = 3, i = 3 일 때,
            # 3번째 노드의 2^3 위의 부모를 가리킨다.
            # 우항에 parent[parent[3][2]][2] 가 오게 되는데 이는
            # 3번째 노드의 2^2 위의 부모의 2^2 위 부모를 가리키기 때문에
            # 결국 2^3 위 부모를 가리키는 것이다.
            parent[j][i] = parent[parent[j][i-1]][i-1]
        
# A 와 B 의 최소 공통 조상을 찾는 함수
def lca(a, b):

    # b가 더 깊도록 설정
    if d[a] > d[b]:
        a, b = b, a

    # 먼저 깊이가 동일하도록 한다.
    for i in range(LOG-1, -1, -1):
        # << 는 비트단위 시프트 연산자로서, 왼쪽으로 1칸 움직인다는 것이다.
        # 아래 식에서는 i >> 1 이기 때문에 오른쪽으로 1비트 움직이기 때문에 반으로 나눈 값이 된다.
        if d[b] - d[a] >= (1 << i):
            b = parent[b][i]
    
    # 부모가 같아지도록
    if a==b:
        return a
        
    for i in range(LOG-1, -1, -1):
        # 조상을 향해 거슬러 올라가기
        if parent[a][i] != parent[b][i]:
            a = parent[a][i]
            b = parent[b][i]
    # 이후에 부모가 찾고자 하는 조상
    return parent[a][0]
    
set_parent()

m = int(input())

for i in range(m):
    a, b = map(int, input().split())
    print(lca(a, b))
```

#### 개선된 LCA 알고리즘 시간 복잡도 분석
- DP 를 이용해서 시간 복잡도를 개선할 수 있다.
- 세그먼트 트리를 이용하는 방법도 존재한다.
  - **오일러 투어 테크닉**과 **세그먼트 트리**를 결합하여 효율적으로 LCA를 찾을 수 있다.
  - 오일러 투어 테크닉은 주어진 트리의 DFS를 수행하면서, 각 노드를 방문한 순서대로 기록한다. 이 때, 방문할 때마다 노드의 깊이를 기록한다.
- 매 쿼리마다 부모를 거슬러 올라가기 위해 $O(logN)$의 복잡도가 필요하다.
- 따라서 모든 쿼리를 처리할 때의 시간 복잡도는 $O(MlogN)$ 이다.

#### 세그먼트 트리를 이용한 LCA 알고리즘
```python
class SegmentTree:
    '''
    SegmentTree

    build : 주어진 배열을 기반으로 세그먼트 트리를 구성
    query : 주어진 범위 내에서 최소값을 찾는다.
    '''
    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self.build(data, 0, 0, self.n - 1)

    def build(self, data, node, start, end):
        if start == end:
            self.tree[node] = data[start]
        else:
            mid = (start + end) // 2
            self.build(data, 2 * node + 1, start, mid)
            self.build(data, 2 * node + 2, mid + 1, end)
            self.tree[node] = min(self.tree[2 * node + 1], self.tree[2 * node + 2])

    def query(self, l, r):
        return self._query(0, 0, self.n - 1, l, r)

    def _query(self, node, start, end, l, r):
        if r < start or end < l:
            return float('inf')
        if l <= start and end <= r:
            return self.tree[node]
        mid = (start + end) // 2
        left_query = self._query(2 * node + 1, start, mid, l, r)
        right_query = self._query(2 * node + 2, mid + 1, end, l, r)
        return min(left_query, right_query)

class LCATree:
    '''
    LCATree 클래스:

    add_edge : 트리에 간선 추가
    dfs : 오일러 투어를 수행하면서, 각 노드를 방문한 순서와 깊이를 기록
    build : 오일러 투어(dfs)를 수행한 후, 깊이와 노드 정보를 기반으로 세그먼트 트리를 구성
    lca : 두 노드의 LCA 찾기
    '''
    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)]
        self.euler = []
        self.depth = []
        self.first = [-1] * n
        self.segment_tree = None

    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    def dfs(self, node, d):
        self.first[node] = len(self.euler)
        self.euler.append(node)
        self.depth.append(d)
        for neighbor in self.graph[node]:
            if self.first[neighbor] == -1:  # 아직 방문하지 않은 경우
                self.dfs(neighbor, d + 1)
                self.euler.append(node)
                self.depth.append(d)

    def build(self, root=0):
        self.dfs(root, 0)
        depth_pairs = [(self.depth[i], self.euler[i]) for i in range(len(self.depth))]
        self.segment_tree = SegmentTree(depth_pairs)

    def lca(self, u, v):
        if self.first[u] > self.first[v]:
            u, v = v, u
        l = self.first[u]
        r = self.first[v]
        _, lca_node = self.segment_tree.query(l, r)
        return lca_node

n = 7
tree = LCATree(n)
edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]

for u, v in edges:
    tree.add_edge(u, v)

tree.build(0)  # 루트를 0으로 설정

print(tree.lca(3, 4))  # 출력: 1
print(tree.lca(3, 5))  # 출력: 0
print(tree.lca(4, 6))  # 출력: 0
print(tree.lca(2, 6))  # 출력: 2
```
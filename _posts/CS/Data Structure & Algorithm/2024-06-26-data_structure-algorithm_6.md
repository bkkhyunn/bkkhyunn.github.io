---
title: "[Algorithm] 6. Binary Search"
excerpt: "이진탐색 알고리즘에 대해 알아보자"

categories: "data_structure-algorithm"
tags:
    - algorithm
    - Binary Search
toc: true  
toc_sticky: true
toc_label: "Contents In Page"
author_profile: true
use_math: true

date: 2024-06-26
---

### 이진 탐색(Binary Search)
- 이진 탐색은 **정렬되어 있는 리스트**에서 특정한 데이터를 빠르게 탐색할 수 있도록 해주는 탐색 알고리즘이다.
- 순차 탐색 : 리스트 안에 있는 특정한 데이터를 찾기 위해 앞에서 데이터를 하나씩 확인하는 방법. 가장 기본적인 형태의 데이터 탐색 알고리즘이다. 선택 정렬에서 매 단계마다 가장 작은 크기의 데이터를 찾는 과정도 이러한 순차 탐색을 이용한 것이다.
- 이진 탐색 : 정렬되어 있는 리스트에서 탐색 범위를 **절반씩 좁혀가며** 데이터를 탐색하는 방법이다. 이진탐색은 이처럼 리스트가 정렬되어 있을 때 사용 가능하다. 정렬되어 있다면 탐색 범위를 절반씩 좁혀 나가면서 데이터를 탐색할 수 있기 때문에 로그 시간의 시간 복잡도를 가진다.
- 이진 탐색을 수행할 때는 **탐색 범위**를 정해줘야 한다. 이 때 탐색 범위를 명시하기 위해서 시작점, 끝점, 중간점을 인덱스로 이용하여 탐색 범위를 설정한다.
- 중간점은 소수점 이하는 제거한다. 예를 들어 10개의 데이터에서 중간점은 인덱스로 4에 해당한다.
- 중간점과 찾고자 하는 값을 비교하여 어떤 값이 더 큰 지를 비교하고, 찾고자하는 값보다 중간점 위치의 값이 더 크다면, 중간점에서부터 오른쪽 값은 확인할 필요가 없다. 왜냐면 이미 정렬된 상태의 리스트이기 때문이다.
- 따라서 끝점을 중간점의 왼쪽으로 옮긴다. 이런 식으로 탐색 범위를 절반씩 줄여 나간다.
- 조건의 만족 여부에 따라서 **탐색 범위를 좁혀서 해결하는 문제**에 활용할 수 있다. 또한 **큰 탐색 범위**가 주어질 때, 가장 먼저 이진 탐색을 떠올리면 좋다.
- 이진 탐색을 수행하면서 탐색 범위를 더 이상 줄이지 못할 때까지 시작점과 끝점을 바꿔가며 수행할 수 있다.
- 이진 탐색의 시간 복잡도
  - 단계마다 탐색 범위를 2로 나누는 것과 동일하므로 연산 횟수는 $log_2N$ 에 비례한다.
  - 예를 들어 초기 데이터 개수가 32개일 때, 이상적으로 1단계를 거치면 16개 가량의 데이터만 남는다. 2단계를 거치면 8개, 3단계를 거치면 4개의 데이터만 남는다.
  - 다시 말해 이진 탐색은 탐색 범위를 절반씩 줄이며, 시간 복잡도는 $O(logN)$ 을 보장한다.

#### 재귀적 구현
```python
# 재귀 함수로 이진 탐색 소스코드 구현
def binary_search(array, target, start, end):
    if start > end:
        return None
    mid = (start + end) // 2
    
    # 찾은 경우 중간점 인덱스 반환. mid 가 찾고자 하는 값의 인덱스라고 할 수 있다.
    if array[mid] == target:
        return mid
    
    # 중간점의 값보다 찾고자 하는 값이 작은 경우 왼쪽 확인
    elif array[mid] > target:
        return binary_search(array, target, start, mid-1)
    
    # 중간점의 값보다 찾고자 하는 값이 큰 경우 오른쪽 확인
    else:
        return binary_search(array, target, mid+1, end)
        
result = binary_search(array, target, 0, n-1)
if result == None:
    print("원소가 존재하지 않습니다.")
else:
    print(result + 1)
```

#### 반복문 구현
```python
# 반복문으로 이진 탐색 소스코드 구현
def binary_search(array, target, start, end):
    while start <= end:
        mid = (start + end) // 2
        
        # 찾은 경우 중간점 인덱스 반환
        if array[mid] == target:
            return mid
        
        # 중간점의 값보다 찾고자 하는 값이 작은 경우 왼쪽 확인
        elif array[mid] > target:
            end = mid - 1
            
        # 중간점의 값보다 찾고자 하는 값이 큰 경우 오른쪽 확인
        else:
            start = mid + 1
    return None
    
result = binary_search(array, target, 0, n-1)
if result == None:
    print("원소가 존재하지 않습니다.")
else:
    print(result + 1)
```
    
#### 파이썬 이진 탐색 라이브러리
- 알아두면 좋은 라이브러리.
- `bisect_left(a, x)` : 정렬된 순서를 유지하면서 배열 a 에 x 를 삽입할 가장 왼쪽 인덱스를 반환. C++ 의 lower bound 와 동일하다.
- `bisect_right(a, x)` : 정렬된 순서를 유지하면서 배열 a 에 x 를 삽입할 가장 오른쪽 인덱스를 반환. C++ 의 upper bound 와 동일하다.

```python
from bisect import bisect_left, bisect_right

a = [1, 2, 4, 4, 8]
x = 4

print(bisect_left(a, x)) # 출력 2
print(bisect_right(a, x)) # 출력 4
```

- 이를 활용해서 값이 특정 범위에 속하는 데이터 개수를 구할 수 있다.

```python
from bisect import bisect_left, bisect_right

# 값이 [left_value, right_value] 인 데이터의 개수를 반환하는 함수
def count_by_range(a, left_value, right_value):
    right_index = bisect_right(a, right_value)
    left_index = bisect_left(a, left_value)
    return right_index - left_index
    
a = [1, 2, 3, 3, 3, 3, 4, 4, 8, 9]

# 값이 4인 데이터 개수 출력
print(count_by_range(a, 4, 4)) # 출력 2

# 값이 -1 부터 3 까지 범위에 있는 데이터 개수 출력
print(count_by_range(a, -1, 3)) # 출력 6
```

#### 파라메트릭 서치 (Parametric Search)
- 이진 탐색을 활용해야 하는 문제가 출제되는 경우 이러한 파라메트릭 서치 유형으로 문제가 출제되는 경우가 많다.
- 파라메트릭 서치란 최적화 문제를 여러번의 결정 문제(’예’ 혹은 ‘아니오’)로 바꾸어 해결하는 기법이다.
- 최적화 문제란 어떤 함수의 값을 가능한 낮추거나 최대한 높이는 등의 문제이다.
- 예를 들어 특정한 조건을 만족하는 가장 알맞은 값을 빠르게 찾는 최적화 문제가 해당한다. 탐색 범위를 좁혀가면서 현재 범위에서 이 조건을 만족하는지 체크하고, 그 여부에 따라서 탐색 범위를 좁혀 가며 가장 알맞은 값을 찾을 수 있다.
- 일반적으로 코딩 테스트에서 파라메트릭 서치 문제는 이진 탐색을 이용하여 해결할 수 있다.
- 중요한 것은 이진탐색의 기준이 되는 값(범위)을 무엇으로 설정할 것인지 떠올려내야 한다.

##### 예제
- 프로그래머스 [입국심사](https://school.programmers.co.kr/learn/courses/30/lessons/43238) 문제
```python
# times 중 가장 오래 걸리는 시간 * 입국 인원 을 하면 가능한 가장 큰 시간이 나온다. 이를 기준으로 이진 탐색을 실시한다.
# 즉 시간을 배열로 보고, 계속해서 중간 시간에 최대 몇 명을 처리할 수 있는지 계산한다.
def solution(n, times):
    times.sort()
    # 시작점, 끝점
    start = 0
    end = times[-1] * n
    
    def binary_search(start, end):
        nonlocal times, n
        
        cnt = 0
        answer = -1

        while start <= end:
            mid = (start + end) // 2
            cnt = 0

            # mid 시간 안에 각 심사관 별로 해당 시간에 몇 명을 처리할 수 있는지
            for time in times:
                cnt += mid // time
            
            # 입국 심사 인원보다 더 많이 처리할 때
            if cnt >= n:
                
                # 정답 최신화
                if answer == -1:
                    answer = mid
                else:
                    answer = min(answer,mid)
                
                # 끝 시간을 중간 시간보다 1 작게 최신화
                end = mid - 1
            
            # 입국 심사 인원보다 더 적게 처리할 때
            else:
                # 시작 시간을 중간 시간보다 1 크게 최신화
                start = mid + 1

        return answer
    
    answer = binary_search(start, end)
    return answer
```
---
title: "[Algorithm] 5. Sorting Algorithm"
excerpt: "정렬 알고리즘에 대해 알아보자"

categories: "data_structure-algorithm"
tags:
    - algorithm
    - Sorting
toc: true  
toc_sticky: true
toc_label: "Contents In Page"
author_profile: true
use_math: true

date: 2024-06-26
---

### 정렬 알고리즘
- 정렬(Sorting)이란 데이터를 특정한 기준에 따라 순서대로 나열하는 것을 말한다.
- 일반적으로 문제 상황에 따라서 적절한 정렬 알고리즘이 공식처럼 사용된다.
- 정렬을 수행하는 프로그램을 작성할 때는 어떠한 방식으로 정렬을 수행할 수 있을지 그 알고리즘을 코드를 이용해서 정확히 표현해야 한다.

#### 선택 정렬(Select Sort)
- 처리되지 않은 데이터 중에서 가장 작은 데이터를 선택해 맨 앞에 있는 데이터와 바꾸는 것을 반복한다.
- 즉 매번 현재 범위에서 가장 작은 데이터를 골라서 제일 앞 쪽으로 보내는 것을 반복한다.
- 맨 마지막에 정렬되지 않은 데이터가 1개가 남았을 때는, 그 데이터를 앞쪽으로 보내봤자 자기 자신의 위치와 동일하기 때문에 마지막 경우에는 처리하지 않아도 전체 데이터가 성공적으로 정렬된다.
- 탐색 범위는 반복할 때마다 줄어들게 되고 매번 가장 작은 원소를 찾기 위해서 탐색 범위만큼 데이터를 하나씩 확인해서 가장 작은 데이터를 찾아야하기 때문에 매번 선형탐색을 수행하는 것과 동일하다.
- 따라서 이중 반복문을 통해서 선택 정렬을 구현할 수 있다.

    ```python
    array = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]

    for i in range(len(array)):
        min_index = i # 가장 작은 원소의 인덱스. 가장 작은 데이터와 위치가 바뀔 인덱스
        for j in range(i+1, len(array)):
            if array[min_index] > array[j]:
                min_index = j
        array[i], array[min_index] = array[min_index], array[i] # 스와프

    # 결과 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ```
- 선택 정렬의 시간 복잡도
  - 선택 정렬은 $N$번 만큼 가장 작은 수를 찾아서 맨 앞으로 보내야 한다.
  - 구현 방식에 따라서 사소한 오차는 있을 수 있지만, 전체 연산 횟수는 $N + (N-1) + (N-2) + … + 2$ 와 같다.
  - 가장 마지막 원소는 정렬을 수행해주지 않아도 괜찮기 때문에 +2 까지 등차수열 형태로 연산횟수가 구성된다.
  - 이는 빅오 표기법에 따라 가장 차수가 높은 항만 고려하여 $O(N^2)$ 로 표현할 수 있다.

#### 삽입 정렬(Insert Sort)
- 처리되지 않은 데이터를 하나씩 골라 적절한 위치에 삽입한다.
- 데이터를 하나씩 확인하면서 이 데이터는 어느 위치에 들어가는 것이 맞을 것인가 매번 계산해서 적절한 위치에 들어갈 수 있게 한다.
- 선택 정렬에 비해 구현 난이도가 높은 편이지만, 일반적으로 더 효율적으로 동작한다.
- 삽입 정렬은 앞쪽에 있는 원소들이 이미 정렬되어 있다고 가정하고, 뒤쪽에 있는 원소를 앞쪽에 있는 원소들의 위치 중에서 한 곳으로 들어가도록 만드는 방식으로 동작한다.
- 따라서 원래 배열의 맨 처음 원소는 정렬된 것이라고 가정하기 때문에 두번째 원소부터 시작한다.
- 차례대로 왼쪽에 있는 데이터(앞쪽에 있는 원소들)와 비교해서 왼쪽 데이터보다 더 작다면 위치를 바꿔주고 그렇지 않다면 그 자리에 머물러 있도록 하면 된다.
- 이 과정들을 계속 반복한다.
    ```python
    array = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]

    for i in range(1, len(array)): # 두번째 원소부터 시작
        for j in range(i, 0, -1): # 인덱스 i 부터 0 까지 1씩 감소하며 반복
            if array[j] < array[j-1]: # 왼쪽 원소와 비교
                array[j], array[j-1] = array[j-1], array[j] # 스와핑
            else: # 자기보다 작거나 같은 데이터를 만나면 그 위치에서 멈춘다.
                break

    # 결과 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ```
- 삽입 정렬의 시간 복잡도
  - 삽입 정렬의 시간 복잡도는 $O(N^2)$ 이며, 선택 정렬과 마찬가지로 반복문이 두 번 중첩되어 사용된다.
  - 삽입 정렬은 현재 리스트의 데이터가 거의 정렬되어 있는 상태라면 매우 빠르게 동작한다.
  - 즉, 최선의 경우 $O(N)$ 의 시간 복잡도를 가진다.
  - 이미 정렬되어 있는 상태에서 다시 삽입 정렬을 수행하면 $O(N)$ 의 시간복잡도를 가진다.

#### 퀵 정렬(Quick Sort)
- 기준 데이터를 설정하고 그 기준보다 큰 데이터와 작은 데이터의 위치를 바꾸는 방법이다.
- 일반적인 상황에서 가장 많이 사용되는 정렬 알고리즘 중 하나이다. 즉 데이터의 특성과 관련없이 표준적으로 사용할 수 있는 정렬 알고리즘 중 하나이다.
- 병합 정렬과 더불어 대부분의 프로그래밍 언어의 정렬 라이브러리의 근간이 되는 알고리즘이다. 대표적으로 C 언어, Java, Python 의 표준 정렬 라이브러리도 퀵정렬 혹은 병합 정렬의 아이디어를 채택한 하이브리드 방식의 정렬 알고리즘을 사용한다.
- 가장 기본적인 퀵 정렬은 첫번째 데이터를 기준 데이터(Pivot)로 설정한다.
- 왼쪽에서부터 피벗값보다 큰 데이터를 고르고, 오른쪽에서부터는 피벗값보다 작은 데이터를 고른다.
- 왼쪽, 오른쪽에서 모두 값이 선택되었으면 그 두 값의 위치를 바꿔준다.
- 왼쪽, 오른쪽으로 가면서 위치가 엇갈리는 경우, 피벗과 오른쪽에서의 고른 작은 데이터의 위치를 서로 변경한다.
- 그러면 피벗값의 위치가 중간으로 들어가게 되고, 왼쪽과 오른쪽으로 분할된다. 피벗값을 중심으로 해서 왼쪽은 피벗값보다 작은 데이터로, 오른쪽은 피벗값보다 큰 데이터로 구성된다.
- 이렇게 하나의 피벗값에 대해서 정렬을 위한 데이터의 범위가 왼쪽과 오른쪽으로 나눠진다는 점에서 이러한 작업을 분할(Divide, Partition)이라고 부른다.

![Untitled](/assets/images/DS&Algorithm/Untitled%202.png){: .align-center}

- 이어서 왼쪽, 오른쪽 데이터들을 대상으로 각각 첫번째 원소를 피벗으로 두고 퀵 정렬을 수행한다.
- 이처럼 퀵 정렬이 수행되는 과정은 재귀적으로 수행되고, 수행될 때마다 정렬을 위한 범위가 점점 좁아지는 형태로 동작한다.
- 계속 재귀적으로 반복하면 전체 데이터에 대해서 정렬이 완료된다.
- 퀵 정렬이 빠른 이유
    - 이상적인 경우로 분할이 절반씩 일어난다면 전체 연산 횟수로 $O(NlogN)$을 기대할 수 있다.
    
    ![Untitled](/assets/images/DS&Algorithm/Untitled%203.png){: .align-center}
    
    - 데이터가 절반씩 줄어들기 때문에 $log_2N$ 만큼의 시간이 걸린다.
    - 퀵 정렬은 평균의 경우 $O(NlogN)$의 시간 복잡도를 가진다.
    - 하지만 최악의 경우 $O(N^2)$의 시간 복잡도를 가진다. 엄밀히 말하면 피벗값을 어떻게 설정하느냐에 따라서 분할이 절반에 가깝게 이루어지지 않고 한쪽 방향으로 편향된 분할이 발생할 수 있다.
    - 첫번째 원소를 피벗으로 삼을 때(단순 구현), 이미 정렬된 배열에 대해서 퀵 정렬을 수행하면 최악의 경우에 해당한다. 즉 피벗이 매번 맨 왼쪽에 위치하고 오른쪽 배열을 모두 훑는다. 그리고 피벗값 자기 자신에서 자기 자신의 위치로 변경된다.
    - 따라서 최악의 경우 분할이 수행되는 횟수가 $N$ 과 비례하고, 분할을 하기 위해서 매번 선형탐색을 수행해야 하기 때문에 전체 시간 복잡도가 $O(N^2)$가 되는 것이다.
- 다양한 프로그래밍 언어에서 표준 정렬 라이브러리를 적용할 때 퀵 정렬을 기반으로 라이브러리가 작성되어 있다면 최악의 경우에도 $O(NlogN)$ 을 보장할 수 있는 형태로 구현한다.
- 따라서 첫번째 원소를 피벗으로 설정하는 방식으로 퀵 정렬을 구현한다면 최악의 경우에 $O(N^2)$ 이 나올 수 있다는 점에 유의해야 한다.
- 표준 라이브러리를 이용하는 경우에는 $O(NlogN)$ 을 항상 보장한다.
    ```python
    array = [5, 7, 9, 8, 3, 1, 6, 2, 4, 8]

    def quick_sort(array, start, end):
        if start >= end: # 원소가 1개인 경우 종료
            return
        pivot = start # 피벗은 첫 번째 원소
        left = start + 1
        right = end
        while left <= right:
            # 피벗보다 큰 데이터를 찾을 때까지 반복 (왼쪽, 선형 탐색)
            while left <= end and array[left] <= array[pivot]:
                left += 1
            # 피벗보다 작은 데이터를 찾을 때까지 반복 (오른쪽, 선형 탐색)
            while right > start and array[right] >= array[pivot]:
                right -= 1
            if left > right: # 엇갈렸다면 작은 데이터와 피벗을 교체
                array[right], array[pivot] = array[pivot], array[right]
            else: # 엇갈리지 않았다면 작은 데이터와 큰 데이터를 교체
                array[left], array[right] = array[right], array[left]
        # 분할 이후 왼쪽 부분과 오른쪽 부분에서 각각 정렬 수행
        quick_sort(array, start, right-1)
        quick_sort(array, right+1, end)
        # 이상적인 경우 위 분할에서 왼쪽과 오른쪽이 균형있게 분할되어 빠르게 동작 기대
        
    quick_sort(array, 0, len(array)-1)
    # 결과: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ```
- 파이썬 장점 살린 방식(list slicing, list comprehension)
    ```python
    array = [5, 7, 9, 8, 3, 1, 6, 2, 4, 8]

    def quick_sort(array):
        # 리스트가 하나 이하의 원소만을 담고 있다면 종료
        if len(array) <= 1:
            return array
        pivot = array[0] # 피벗은 첫번째 원소
        tail = array[1:] # 피벗을 제외한 리스트
        
        left_side = [x for x in tail if x <= pivot] # 분할된 왼쪽 부분
        right_side = [x for x in tail if x > pivot] # 분할된 오른쪽 부분
        
        # 분할 이후 왼쪽 부분과 오른쪽 부분에서 각각 정렬 수행하고 전체 리스트 반환
        return quick_sort(left_side) + [pivot] + quick_sort(right_side)
        
    # 결과: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ```
        
#### 병합 정렬(Merging Sort)
- 일반적인 방법으로 구현했을 때 이 정렬은 안정 정렬에 속하며, 퀵 정렬과 같이 분할 정복 알고리즘의 하나이다.
- 분할 정복(divide and conquer) 방법
  - 문제를 작은 2개의 문제로 분리하고 각각을 해결한 다음, 결과를 모아서 원래의 문제를 해결하는 전략이다.
  - 분할 정복 방법은 대개 순환 호출을 이용하여 구현한다.
  - 과정
      1. 리스트의 길이가 0 또는 1이면 이미 정렬된 것으로 본다. 그렇지 않은 경우에는
      2. 정렬되지 않은 리스트를 절반으로 잘라 비슷한 크기의 두 부분 리스트로 나눈다.
      3. 각 부분 리스트를 재귀적으로 병합 정렬을 이용해 정렬한다.
      4. 두 부분 리스트를 다시 하나의 정렬된 리스트로 병합한다.
- 하나의 리스트를 두 개의 균등한 크기로 분할하고 분할된 부분 리스트를 정렬한 다음, 두 개의 정렬된 부분 리스트를 합하여 전체가 정렬된 리스트가 되게 하는 방법이다.
- 병합 정렬은 다음의 단계들로 이루어진다.
  - 분할(Divide): 입력 배열을 같은 크기의 2개의 부분 배열로 분할한다.
  - 정복(Conquer): 부분 배열을 정렬한다. 부분 배열의 크기가 충분히 작지 않으면 순환 호출을 이용하여 다시 분할 정복 방법을 적용한다.
  - 결합(Combine): 정렬된 부분 배열들을 하나의 배열에 합병한다.
- 병합 정렬의 과정
  - 추가적인 리스트가 필요하다.
  - 각 부분 배열을 정렬할 때도 병합 정렬을 순환적으로 호출하여 적용한다.
  - 병합 정렬에서 실제로 정렬이 이루어지는 시점은 2개의 리스트를 병합(merge)하는 단계다.
    
    ![Untitled](/assets/images/DS&Algorithm/Untitled%204.png){: .align-center}
    
    ```python
    def merge_sort(arr):
        if len(arr) > 1:
            mid = len(arr) // 2  # 중간 지점을 계산
            left_half = arr[:mid]  # 왼쪽 부분 배열
            right_half = arr[mid:]  # 오른쪽 부분 배열
    
            merge_sort(left_half)  # 왼쪽 부분 배열을 재귀적으로 정렬
            merge_sort(right_half)  # 오른쪽 부분 배열을 재귀적으로 정렬
            
            # i: left_half 배열의 현재 위치를 나타내는 인덱스
            # j: right_half 배열의 현재 위치를 나타내는 인덱스
            # k: 원래 배열 (arr)의 현재 위치를 나타내는 인덱스
            i = j = k = 0
    
            # 좌우 부분 배열을 병합
            while i < len(left_half) and j < len(right_half):
                if left_half[i] < right_half[j]:
                    arr[k] = left_half[i]
                    i += 1
                else:
                    arr[k] = right_half[j]
                    j += 1
                k += 1
    
            # 왼쪽 부분 배열에 남은 요소들 복사
            while i < len(left_half):
                arr[k] = left_half[i]
                i += 1
                k += 1
    
            # 오른쪽 부분 배열에 남은 요소들 복사
            while j < len(right_half):
                arr[k] = right_half[j]
                j += 1
                k += 1
    
    # 예제
    arr = [38, 27, 43, 3, 9, 82, 10]
    print("Original array:", arr)
    merge_sort(arr)
    print("Sorted array:", arr)
    ```
- 병합 정렬 시간복잡도 분석
  - 병합 정렬의 시간 복잡도는 $O(NlogN)$ 이다.
  - 분할 연산에서 $O(logN)$ 이 걸린다.
  - 이후 분할된 상태에서 병합을 할 때 각 요소들을 비교하면서 정렬하기 때문에 비교 연산이 진행된다. 이 비교 연산을 n 번 수행한다. 따라서 $O(NlogN)$ 이다.

#### 계수 정렬(Counting Sort)
- 특정한 조건이 부합할 때만 사용할 수 있지만 매우 빠르게 동작하는 정렬 알고리즘이다.
- 계수 정렬은 데이터의 크기 범위가 제한되어 정수 형태로 표현할 수 있을 때 사용 가능하다.
- 데이터의 개수가 $N$, 데이터(양수) 중 최댓값이 $K$ 일 때 최악의 경우에도 수행시간 $O(N+K)$ 를 보장한다.
- 가장 작은 데이터부터 가장 큰 데이터까지의 범위가 모두 담길 수 있도록 리스트를 생성한다.
- 새로 생긴 리스트(계수 리스트)는 각 인덱스가 데이터의 값에 해당한다. 즉 각 데이터가 총 몇 번 등장했는지를 세는 것이다. 개수 리스트의 3번 인덱스는 정렬할 데이터에서 3이 몇 번 등장했는지를 담고 있다.
- 배열에서 특정 인덱스에 접근할 때 알고리즘 상으로 상수 시간의 접근이 가능하다. 그래서 결과적으로 데이터를 하나씩 확인하면서, 그 데이터와 동일한 인덱스에 카운트 값을 1씩 증가시키면서 해당 데이터가 몇 번 등장했는지를 체크하는 것이다.
- 각 원소의 카운트를 모두 구한 뒤에는 실제로 정렬 수행 결과를 확인할 수 있다. 결과를 확인할 때는 개수 리스트의 첫번째 데이터부터 하나씩 그 값만큼 반복하여 해당 인덱스를 출력한다.
- 그렇게 되면 전체 출력결과는 정렬이 수행된 결과와 동일하다.
- 계수 정렬은 각 데이터가 몇 번씩 등장했는지를 세는 방식으로 동작하는 정렬 알고리즘이다. 가장 작은 데이터부터 가장 큰 데이터까지의 모든 범위를 포함할 수 있는 크기의 배열을 만들어야 하기 때문에 상대적으로 공간 복잡도가 높지만 퀵 정렬과 비교했을 때 조건만 만족한다면 더 빠르게 동작한다.

    ```python
    # 모든 원소의 값이 0보다 크거나 같다고 가정
    array = [7, 5, 9, 0, 3, 1, 6, 2, 9, 1, 4, 8, 0, 5, 2]
    # 모든 범위를 포함하는 리스트 선언(모든 값은 0으로 초기화)
    count = [0] * (max(array)+1)

    for i in range(len(array)):
        # count 배열에 각 데이터가 몇번씩 등장했는지를 기록
        count[array[i]] += 1 # 각 데이터에 해당하는 인덱스의 값 증가

    for i in range(len(count)):
        for j in range(count[i]):
            print(i, end=' ') # 띄어쓰기를 구분으로 등장한 횟수만큼 인덱스 출력
            
    # 결과: 0 0 1 1 2 2 3 4 5 5 6 7 8 9 9
    ```
- 계수 정렬의 시간 복잡도
  - 계수 정렬의 시간 복잡도와 공간 복잡도는 모두 $O(N+K)$ 이다.
    - 각 데이터가 몇번씩 등장했는지를 기록할 때 $O(N)$, 원소 중 가장 큰 값인 K 만큼 인덱스를 확인하여 기록된 값 만큼 인덱스를 출력할 때 $O(K)$, 그 안쪽 반복문은 $O(N)$
  - 계수 정렬은 때에 따라서 심각한 비효율성을 초래할 수 있다.
  - 예를 들어 데이터가 0 과 999,999로 단 2개만 존재하는 경우를 들 수 있다. 따라서 데이터의 범위가 너무 크다면 계수 정렬은 사용하기 어렵다는 특징이 있다.
  - 계수 정렬은 동일한 값을 가지는 데이터가 여러 개 등장할 때 효과적으로 사용할 수 있다.
  - 성적의 경우 100점을 맞은 학생이 여러 명일 수 있기 때문에 계수 정렬이 효과적이다.

#### 정렬 알고리즘 비교
- 대부분의 프로그래밍 언어에서 지원하는 표준 정렬 라이브러리는 최악의 경우에도 $O(NlogN)$을 보장하도록 설계되어 있다.
- 파이썬의 경우 병합 정렬을 기반으로 하는 하이브리드 방식의 정렬 알고리즘을 사용하고 있어 최악의 경우에도 $O(NlogN)$ 의 시간 복잡도를 보장한다.
 
    | 정렬 알고리즘 | 평균 시간 복잡도 | 공간 복잡도 | 특징 |
    | --- | --- | --- | --- |
    | 선택 정렬 | $O(N^2)$ | $O(1)$ | 아이디어가 매우 간단하다. |
    | 삽입 정렬 | $O(N^2)$ | $O(1)$ | 데이터가 거의 정렬되어 있을 때는 가장 빠르다.($O(N)$) |
    | 퀵 정렬 | $O(N\log N)$ | $O(\log N)$ | 대부분의 경우에 가장 적합하며, 충분히 빠르다. 구현 방식에 따라서 최악의 경우 시간 복잡도가 $O(N^2)$ 이 나올 수 있다. 예를 들어 거의 정렬된 배열에서 피벗을 첫번째 원소로 설정하면 최악의 경우가 될 수 있다. |
    | 병합 정렬 | $O(N\log N)$ | $O(N)$ | 안정 정렬 알고리즘으로, 항상 $O(N\log N)$ 의 시간 복잡도를 가지며, 추가적인 메모리 공간을 필요로 한다. |
    | 계수 정렬 | $O(N+K)$ | $O(N+K)$ | 데이터의 크기가 한정되어 있는 경우에만 사용이 가능하지만 매우 빠르게 동작한다. |
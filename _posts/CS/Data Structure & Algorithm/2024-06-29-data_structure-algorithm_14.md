---
title: "[Algorithm] 14. Trie"
excerpt: "트라이에 대해 알아보자"

categories: "data_structure-algorithm"
tags:
    - algorithm
    - Tree
    - Trie
toc: true  
toc_sticky: true
toc_label: "Contents In Page"
author_profile: true
use_math: true

date: 2024-06-29 13:30
---

### 트라이(Trie) 자료구조
- Trie 란, 입력되는 문자열을 Tree 형식으로 만들어 보다 빠르게 문자열 검색이 가능한 자료구조다. 이러한 특징으로 자동완성 기능, 사전 검색 등 문자열을 탐색하는데 특화되어있는 자료구조라고 한다.
- **Prefix Tree** 라고도 부른다.
- 보통 배열에서 문자열이 존재하는지 확인하기 위해서는 $O(n)$이라는 시간이 걸리는데 Trie 알고리즘을 사용하면 $O(m)$(m은 제일 긴 문자열의 길이)이라는 짧은 시간이 소요되기 때문에 엄청 효율적이다.
- 빠른 시간복잡도 덕분에 검색엔진 사이트에서 제공하는 자동 완성 및 검색어 추천 기능에서 Trie 알고리즘을 사용한다.
- 빠르게 탐색이 가능하다는 장점이 있지만 각 노드에서 자식들에 대한 포인터들을 배열로 모두 저장하고 있다는 점에서 저장 공간의 크기가 크다는 단점도 있다.

#### 트라이 동작과정
- 트라이는 집합에 포함된 문자열의 접두사들에 대응되는 노드들이 서로 연결된 트리다.
- `{”rebro”, “replay”, “hi”, “high”, “algo”}` 를 가지고 트라이를 구성해보자.
- 아래 그림처럼 한 문자열에서 다음에 나오는 문자가 현재 문자의 자식 노드가 되고, 빨간색으로 나타낸 노드는 문자열의 끝을 의미한다.

![Untitled](/assets/images/DS&Algorithm/trie.png){: .align-center}*이미지 출처: https://velog.io/@jeongbeom4693/Java-Trie*

- 트라이 구조가 문자열을 탐색하기 위한 자료구조이므로, 문자열을 탐색하기 위해서는 다음 글자에 해당하는 노드가 연결되어 있는지, 연결되어 있다면 그 노드를 타고 계속해서 따라가야한다.
- 문자열의 끝에 도달했을 때 즉, 빨간색 노드가 나왓을 때 루트에서부터 내려가는 경로에서 만나는 글자들을 모으면 찾고자 하는 문자열을 확인할 수 있다.
- 동작과정
  1. 문자열을 루트에서부터 시작하여 한 글자씩 탐색한다.
  2. 현재 글자에 해당하는 노드가 존재하면 그 노드로 이동한다.
  3. 더 이상 탐색할 수 없을 때까지 1, 2 과정을 반복한다.
  4. 탐색이 끝나면 마지막 노드가 단어의 끝인지 여부를 확인한다. 단어의 끝이라면 해당 단어가 트라이에 존재한다는 것이다.

#### 트라이 구현
```python
class TrieNode:
    def __init__(self):
        self.children = {}  # 자식 노드들을 저장하는 딕셔너리
        self.is_end_of_word = False  # 해당 노드가 단어의 끝인지 여부를 나타내는 플래그

class Trie:
    def __init__(self):
        self.root = TrieNode()  # 루트 노드 생성

    def insert(self, word):
        """
        주어진 단어를 트라이에 삽입하는 메서드
        """
        node = self.root
        for char in word:
            # 자식 노드에 문자가 없으면 새로운 노드를 생성하여 추가
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        # 단어의 끝을 표시
        node.is_end_of_word = True

    def search(self, word):
        """
        주어진 단어가 트라이에 있는지 검색하는 메서드
        """
        node = self.root
        for char in word:
            if char not in node.children:
                return False  # 해당 문자가 없으면 검색 실패
            node = node.children[char]
        # 마지막 노드가 단어의 끝인지 여부를 반환
        return node.is_end_of_word

    def starts_with(self, prefix):
        """
        주어진 접두사로 시작하는 단어가 있는지 검색하는 메서드
        """
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False  # 해당 접두사가 없으면 검색 실패
            node = node.children[char]
        return True  # 모든 문자가 존재하면 접두사가 존재함을 반환

# 테스트
trie = Trie()
words = ["apple", "banana", "orange", "app", "applesauce"]
for word in words:
    trie.insert(word)

# 검색 테스트
print("Search 'apple':", trie.search("apple"))  # True
print("Search 'apples':", trie.search("apples"))  # False
print("Search 'ora':", trie.search("ora"))  # False

# 접두사 검색 테스트
print("Starts with 'app':", trie.starts_with("app"))  # True
print("Starts with 'ora':", trie.starts_with("ora"))  # True
print("Starts with 'bananas':", trie.starts_with("bananas"))  # False
```
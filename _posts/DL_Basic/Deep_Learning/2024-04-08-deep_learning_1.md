---
title: "[Deep Learning] 1. Linear Neural Networks for Regression"
excerpt: "선형 회귀(Linear Regression)에 대해 알아보고 이후 Neural Network 로 발전하는 과정을 보자"

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

## 회귀(Regression)
- 회귀 문제는 주어진 데이터 $x$에 해당하는 실제 값으로 주어지는 타겟 $y$ 를 예측하는 과제이다.
- 회귀 문제는 현실에서 많이 보이는 문제이다. 예를 들면, 주택 가격, 기온, 판매량 등과 같은 **연속된 값을 예측**하는 문제들을 들 수 있다.
- 이는 결과 값이 이미지 **분류**와 같이 과일의 종류를 예측하는 **이산적인(discrete) 분류 문제(classification)**와는 다르다.
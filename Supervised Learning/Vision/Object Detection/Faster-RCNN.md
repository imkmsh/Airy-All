# Object Detection
![](../../../img/1%202%20stage%20detector.png)
## 2-Stage Detector
물체의 위치를 찾는 문제(localization)과 분류(classification) 문제를 순차적으로 해결

이미지 안에서 물체가 있을 법한 위치를 찾아 나열하고(Region proposals) 각각의 위치에 대해 feature extract, classify, 위치에 대한 정보 조정(bbox 조정)

 ex) RCNN

## 1-Stage Detector
물체의 위치를 찾는 문제(localization)과 분류(classification) 문제를 한 번에 해결
빠른 대신 정확도 낮음

ex) YOLO

# Region Proposal

## Sliding window

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbHiDLF%2Fbtq7SMUlOw9%2FOjrUnQHvHvi3cijLbSDziK%2Fimg.gif)

이미지에서 다양한 형태의 window를 슬라이딩하며 물체가 존재하는지 확인

너무 많은 영역에 대하여 확인해야 한다는 단점, feature map이 아니라 입력 이미지에 대해서 cpu 장치를 이용해 sliding window를 진행하게 되면 넓은 input space 상에서 모두 탐색하게 되어 느릴 수 있다는 단점

## Selective Search

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FoXEM3%2Fbtq7Nyct3hD%2FCKJMxE5jdzbgtG02wm7b9K%2Fimg.png)

인접한 영역(region) 끼리 유사성을 측정해 큰 영역으로 차례대로 통합, 대표적으로 RCNN과 Fast RCNN에서

# NMS

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FkBBSQ%2Fbtq7QwrSQ1m%2FzTJz4h42YpFyAjQ5BjzKO1%2Fimg.png)

같은 class끼리 IoU가 특정 임계점(threshold) 이상일 때 낮은 confidence box를 제거하여 중복된 bounding box 제거


# Bounding Box regression

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbB6gzG%2Fbtq7UzHIZes%2FimEBITYkUnHFIFzOLboVa1%2Fimg.png)

Localization 성능을 높이기 위해 좌표(x,y,w,h)에 대해서 학습을 진행하면서 linear regression으로 좌표의 위치를 조정 및 예측


# 정확도 측정

## Precision & Recall
|채점|결과|이름|
|---|---|---|
|O|Positive|TP(True Positive) 있다고 올바르게 판단|
|O|Negative|TN(True Negative) 없다고 올바르게 판단|
|X|Positive|FP(False Positive) 있다고 틀리게 판단|
|X|Negative|FN(False Negative) 없다고 틀리게 판단|

1. Precision: 올바르게 탐지한 물체 / 모델이 탐지한 물체 = TP / TP + FP

2. Recall: 올바르게 탐지한 물체 / 실제 정답 물체 = TP / TP + FN

일반적으로 Precision과 Recall은 반비례 관계를 가지기 때문에 Recall에 따른 Precision값을 고려하기 위하여 Average Precision(AP)으로 측정

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F6qI1E%2Fbtq7RKCuxgQ%2FsocA2omNKGSIlYA8J3jjU0%2Fimg.png)

단조 감소 그래프로 표현, 빨간색 영역을 채워넣고 보라색과 빨간색을 합친 영역 넓이 계산

**mAP@0.5**: 정답과 예측의 IoU가 50% 이상일 때 정답으로 판정

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fd0p9Ux%2Fbtq7QQpOilb%2FyojtOs8hc4NCY38CY1OwCK%2Fimg.png)

# R-CNN 계열

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcXnaWt%2Fbtq7RKIX1vR%2FiPk7Ih4bkVf7DQspWXnnrk%2Fimg.png)

## R-CNN: Regions with CNN features (CVPR2014)  

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb6Qg6r%2Fbtq7UPKkfu5%2FopqQwOegZwxgUvcOWKixtk%2Fimg.png)

CPU에서 Selective Search
→ 이 과정에서 물체가 있을 법한 위치를 2000개 찾고
→ 2000개 부분을 crop하고 정사각형으로 resizing해서 다 똑같은 크기로
→ **개별적으로** CNN network를 거쳐서 feature vector 추출
→ SVM을 통해 classification
→ Regressor를 통해 bbox의 정확한 위치 예측

1. Selective Search를 이용해 2000개의 ROI(Region of Interest)를 추출한다. (on CPU )

2. 각 ROI에 대하여 warping을 수행하여 동일한 크기의 입력 이미지로 변경한다.

3. Warped image를 CNN에 넣어서 이미지 feature를 추출한다.

4. 해당 feature를 SVM에 넣어 class의 분류 결과를 얻는다. (binary SVM Classifier[Yes or No] 모델 사용)

5. 해당 feature를 Regressor에 넣어 위치(bounding box)를 예측한다.
   
### 한계

1. 입력 이미지에 대해서 CPU 기반의 Selective Search를 진행해야하므로 많은 시간이 소요됨

2. 전체 아키텍처에서 SVM, Regressor 모듈이 CNN과 분류되어 있음 따라서 SVM과 Regressor의 결과를 통해서 CNN을 업데이트 할 수 없음 따라서 이는 End-to-End 방식으로 학습할 수가 없음

3. 모든 ROI를 CNN에 넣고 학습해야하기 때문에 ROI의 개수만큼 즉, 본 논문에선 2000번의 CNN 연산이 필요 따라서 이로 인해 많은 시간이 소요됨


## Fast R-CNN (ICCV 2015)  
CPU에서 Selective Search를 통해 Region Proposal을 찾음, CNN을 거쳐 Feature vector 추출, (Selective Search된 이미지를 모두 CNN network를 거치지 않고, 단 한 번만 거침, 속도 향상의 원인) ROI pooling을 통해 각각의 Region에 대해 Featur extract(CNN 구조 상 객체에 대한 이미지의 위치 정보가 담겨져 있기 떄문에 가능), Softmax layer 거쳐서 각 class의 probability 구하고 이를 이용해 classification  
   

## Faster-RCNN  
CPU에서 Selective Search하는 속도 단점을 개선하기 위해 **GPU**에서 연산, 이를 위해 **Region Proposal Network(RPN)** 도입, Feature map을 보고 어디에 물체가 있을지 예측, RPN에서 한 번 forwarding하면 어디에 물체가 있을 법한지 예측할 수 있기 떄문에 더 빠르고 정확하게 모델 동작, 나머지는 Fast RCNN과 동일하게 Softmax
   

|이름|-|내용|
|:---:|:---:|---|
|R-CNN|장점|CNN을 이용해 각 Region의 클래스를 분류할 수 있다|
|R-CNN|단점|전체 framework를 End-to-End 방식으로 학습할 수 없다 / Global Optimal Solution을 찾을 수 없다|
|Fast R-CNN|장점|Feature Extraction, ROI poolingm Region Classification,Bouding Box Regression 단계를 모두 End-to-End로 묶어서 학습될 수 있다|
|Fast R-CNN|단점| Selective Search는 CPU에서 수행되므로 속도가 느리다|
|Faster R-CNN|장점|**RPN**을 제안하여, 전체 프레임워크를 End-to-End로 학습할 수 있다|
|Faster R-CNN|단점|여전히 많은 컴포넌트로 구성되며, Region Classification 단계에서 각 특징 벡터(feature vector)는 개별적으로 FC layer로 Forward 된다|
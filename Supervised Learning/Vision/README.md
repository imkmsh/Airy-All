# LeNet5

기존의 Fully-Connected Neural Network가 가지고 있는 한계를 이해하고 이것을 개선하기 위해 연구 시작

![기본 구조(그림)](https://m.blog.naver.com/laonple/221218707503?view=img_1)

input layer, 3 conv layer, 2 subsampling layer, full-connected

input - C1 - S2 - C3 - S4 - C5 - F6 - output

Every layer's activation function: tanh

- C1 Layer  

  input(32x32) → convolution with 6 filters(5x5) → 6 28x28 feature map  
  why 6 feature map: each filter makes each feature map  
  why the size decreased from 32 to 28: no padding
  

- S2 Layer  
2 x 2 필터를 stride 2로 설정
  

- C3 Layer  
6장의 14 x 14 feature map으로부터 16장의 10 x 10 feature map을 산출
  

- S4 Layer  
10 x 10 feature map 영상을 5 x 5로
  

- C5 Layer  
16개의 5 x 5 영상을 받아 kernel 크기의 convolution을 수행하기 때문에 출력은 1 x 1 크기의 feature map   
  

- F6 Layer  
Fully Connected, 84개의 unit에 연결
  

# AlexNet

영상 데이터 베이스를 기반으로 한 인식 대회 'ILSVRC 2012'에서 우승한 CNN 구조   


1. ReLU 함수  
   : LeNet-5에서 사용던 tanh 대신 ReLU, 정확도를 유지하면서 tanh을 사용하는 것보다 6배 빠름
   

2. dropout  
   : over-fitting을 막기 위해서, train에 적용되는 것이고, test에는 모든 뉴런 사용 
   

3. overlapping pooling  
   : CNN에서 pooling의 역할은 feature map size 줄이기  
   LeNet-5은 average pooling, AlexNet은 max pooling  
   LeNet-5의 경우 풀링 커널이 움직이는 보폭인 stride를 커널 사이즈보다 작게 하는 overlapping pooling, 따라서 LeNet-5는 non-overlapping 평균 풀링을 사용한 것, AlexNet은 overlapping 최대 풀링  
   overlapping 풀링을 하면 풀링 커널이 중첩, non-overlapping 풀링을 하면 중첩없이, overlapping 풀링이 top-1, top-5 에러율을 줄이는데 효과적
   

4. local response normalization (LRN)  
   : 신경생물학 lateral inhibition 개념, 활성화된 뉴런이 주변 이웃 뉴런들을 억누르는 현상  
   lateral inhibition 현상을 모델링한 것이 local response normalization, 강하게 활성화된 뉴런의 주변 이웃들에 대해서 normalization
   

5. data augmentation  
   : over-fitting 막기 위해, 데이터의 양을 늘리는 방법  
   하나의 이미지로 여러 장의 비슷한 이미지를 만들어 냄, 좌우 반전, 227 x 227 x 3보다 큰 이미지를 조금씩 다르게 잘라서 227 x 227 x 3으로

2개의 GPU를 기반으로 한 병렬 구조

![기본 구조(그림)](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F99FEB93C5C80B5192E), input size가 224로 되어있는데 227이 맞음

- C1 Layer  
96개의 11 x 11 x 3 커널로 입력 영상을 컨볼루션, stride 4, 96장의 55 x 55 feature map 산출, ReLU,  3 x 3 overlapping max pooling이 stride 2로, 96장의 27 x 27 feature map, 수렴 속도를 높이기 위해 local response normalization(→ 현재는 batch normalization을 사용한다고 함), feature map size 유지


- C2 Layer  
256개의 5 x 5 x 48 커널로 컨볼루션, stride 1, padding 2, 27 x 27 x 256 feature map 산출, ReLU, 3 x 3 overlapping max pooling을 stride 2로, 13 x 13 x 256 feature map, local response normalization, feature map size 유지
  

- C3 Layer  
384개의 3 x 3 x 256 커널로 컨볼루션, stride 1, padding 1, 13 x 13 x 384 feature map 산출, ReLU
  

- C4 Layer  
384개의 3 x 3 x 192 커널로 컨볼루션, stride 1, padding 1, 13 x 13 x 384 feature map 산출, ReLU
  

- C5 Layer  
256개의 3 x 3 x 192 커널로 컨볼루션, stride 1, padding 1, 13 x 13 x 256 feature map 산출, ReLU, 3 x 3 overlapping max pooling을 stride 2로, 6 x 6 x 256 feature map
  

- F6 Layer  
6 x 6 x 256 feature map flatten, 6 x 6 x 256 = 9216 dim vector 산출, 4096개 뉴런과 fully connected, ReLU
  

- F7 Layer  
4096개의 뉴런, 전 단계의 4096개 뉴런과 fully connected, ReLU
  

- F8 Layer  
1000개의 뉴런, 전 단계의 4096개 뉴런과 fully connected, softmax

# VGGNet 

Alexnet이 VGGNet으로 발전할 때,
1. 수용영역은 같은데 파라미터 수 줄일 수 있음
   - ex)   
      11 x 11 x D filter N개, 계층 1개   
      11 x 11 x D x N개의 파라미터 개수 = 121DN
      3 x 3 x D filter, 계층 5개  
        3 x 3 x D x N + 4(3 x 3 x N x N) = 9DN + 36N^2  
      N < 3.6D 이면 파라미터 수 감소  
2. 비선형성 증가
3. ReLU로 gradient 소멸 해결
   Sigmoid는 곱할수록 1/4, ReLu는 기울기 1
4. 마지막에 Fully Connected Layer 대신 Fully Convolutional Layer을 제안 (필터 모양을 아예 똑같게)
5. dropout으로 정규화

# GoogleLeNet

![googlelenet](C:\Users\82106\Desktop\CSE\Airy-All\img\googlelenet.png)

Alexnet이 GoogleLeNet으로 발전할 때,
- Inception Block
  - 1 x 1 conv
  - 1 x 1 conv 3 x 3 conv
  - 1 x 1 conv 5 x 5 conv
  - 3 x 3 pool 1 x 1 conv

4 Network를 거쳐서 병렬 연결, 1 x 1 x D x (D/2) -> Depth가 너무 커지는 현상 방지, 다른 정보는 유지하고 채널 깊이만 감소, 병목현상 해결

1. 매개변수 개수가 VGG보다 적음
2. Filter_size 중요도를 스스로 선택
3. Concat으로 비선형성 증가

# Resnet

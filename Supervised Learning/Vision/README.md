# LeNet5

기존의 Fully-Connected Neural Network가 가지고 있는 한계를 이해하고 이것을 개선하기 위해 연구 시작

- 기본 구조

input layer, 3 conv layer, 2 subsampling layer, full-connected

input - C1 - S1 - C2 - S2 - C3 - F - output

Every layer's activation function: tanh

- C1 Layer  
  input(32x32) → convolution with 6 filters(5x5) → 6 28x28 feature map  
  why 6 feature map: each filter makes each feature map  
  why the size decreased from 32 to 28: no padding
  


# AlexNet
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

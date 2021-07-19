# RNN과 LSTM

#RNN

Recurrent Neural Network   
: Model, process input and output in sequences

Hidden: ht = tanh(Wx xt + Wh h(t-1) + b)   
Output: yt = f(Wy ht + b)

#LSTM

Long Shor-Term Memory  
: Variation to complement Vanilla RNN

the problem of Long-Term Dependencies

1. forget gate: 과거 정보를 잊기 위한 게이트
2. input gate: 현재 정보를 기억하기 위한 게이트, it 의 범위는 0~1, gt의 범위는 -1~1이기 때문에 각각 강도와 방향

ft = σ(Wxh_fxt+Whh_fht−1+bh_f)
it = σ(Wxh_ixt+Whh_iht−1+bh_i)
ot = σ(Wxh_oxt+Whh_oht−1+bh_o)
gt = tanh(Wxh_gxt+Whh_ght−1+bh_g)
ct = ft⊙ct−1+it⊙gt
ht = ot⊙tanh(ct)

https://blog.kakaocdn.net/dn/Fcvmn/btqwjvCGq13/hE2mGMQ2HRKBOcIYSx7CSK/img.png

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import pandas as pd
import sentencepiece as spm
import random

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp

import matplotlib.pyplot as plt
import pickle

#encoder vocab
sp_e = spm.SentencePieceProcessor()
vocab_file = "translate_english.model"
sp_e.load(vocab_file)

#decoder_vocab
sp = spm.SentencePieceProcessor()
vocab_file = "translate_french.model"
sp.load(vocab_file)

"""
encoder sentence max length = max_encoder
decoder sentence max length = max_decoder
"""

try:
    with open('encoder_index.txt', 'rb') as f:
        while True:
            try:
                encoder_index = pickle.load(f)
            except EOFError:
                break
    max_encoder = len(encoder_index[0])
except FileNotFoundError:
    #transformer translate
    en = pd.read_fwf('baseline-1M_train.en', header = None, keep_default_na = False)
    en = en.loc[:,0]
    fr = pd.read_fwf('baseline-1M_train.fr', header = None, keep_default_na = False)
    fr = fr.loc[:,0]

    #encoder, decoder input, output
    encoder = list(en)
    decoder = list(fr)

    #data to embedding
    max_encoder = max([len(sp_e.encode(line, out_type=int)) for line in encoder])
    max_decoder = max([len(sp.encode(line, out_type=int)) for line in decoder]) + 1

    encoder_index = []
    for str in encoder:
                temp = sp_e.encode(str, out_type=int)
                i = len(temp)
                while i < max_encoder:
                    temp.append(0)
                    i += 1
                encoder_index.append(temp)
    with open('encoder_index.txt', 'wb') as f:
        pickle.dump(encoder_index, f)

try:
    with open('decoder_input_index.txt', 'rb') as f:
        while True:
            try:
                decoder_input_index = pickle.load(f)
            except EOFError:
                break
    max_decoder = len(decoder_input_index[0])
except FileNotFoundError:
    decoder_input_index=[]
    sp.SetEncodeExtraOptions('bos')
    for str in decoder:
        temp = sp.encode(str, out_type=int)
        i = len(temp)
        while i < max_decoder:
            temp.append(0)
            i += 1
        decoder_input_index.append(temp)
    with open('decoder_input_index.txt', 'wb') as f:
        pickle.dump(decoder_input_index, f)

try:
    with open('decoder_output_index.txt', 'rb') as f:
        while True:
            try:
               decoder_output_index = pickle.load(f)
            except EOFError:
                break
except FileNotFoundError:
    decoder_output_index=[]
    sp.SetEncodeExtraOptions('eos')
    for str in decoder:
        temp = sp.encode(str, out_type=int)
        i = len(temp)
        while i < max_decoder:
            temp.append(0)
            i += 1
        decoder_output_index.append(temp)
    with open('decoder_output_index.txt', 'wb') as f:
        pickle.dump(decoder_output_index, f)


class CustomDataset(Dataset):
    def __init__(self, x, y, z):
        self.x_data = x
        self.y_data = y
        self.z_data = z

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.tensor(self.x_data[idx])
        y = torch.tensor(self.y_data[idx])
        z = torch.tensor(self.z_data[idx])
        return x, y, z


class PositionalEncoding(nn.Module):
    #position = sequence length, d_model = embedding dimension
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
        
    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000, (2 * (i // 2)) / d_model)
        return torch.Tensor(position * angles)

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position = np.array([i for i in range(int(position))], dtype = np.float32).reshape(-1, 1),
            i = np.array([i for i in range(int(d_model))], dtype = np.float32).reshape(1, -1),
            d_model=d_model)

        # 배열의 짝수 인덱스(2i)에는 사인 함수 적용
        sines = torch.sin(angle_rads[:, 0::2])

        # 배열의 홀수 인덱스(2i+1)에는 코사인 함수 적용
        cosines = torch.cos(angle_rads[:, 1::2])

        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines

        pos_encoding = angle_rads
        pos_encoding = pos_encoding.unsqueeze(0)

        return pos_encoding.cuda()

    def forward(self, inputs):
        return self.pos_encoding + inputs

class scaled_dot_product_attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim = -1)
    
    def forward(self, query, key, value, mask = None):
        # query 크기 : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        # key 크기 : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
        # value 크기 : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
        # padding_mask : (batch_size, 1, 1, key의 문장 길이)

        # Q와 K의 곱. 어텐션 스코어 행렬.
        matmul_qk = torch.matmul(query, torch.transpose(key, -2, -1))

        # 스케일링
        # dk의 루트값으로 나눠준다.
        dk = torch.Tensor([key.shape[-1]]).cuda()
        logits = matmul_qk / torch.sqrt(dk)

        # 마스킹. 어텐션 스코어 행렬의 마스킹 할 위치에 매우 작은 음수값을 넣는다.
        # 매우 작은 값이므로 소프트맥스 함수를 지나면 행렬의 해당 위치의 값은 0이 된다.
        if mask is not None:
            logits += (mask * -1e9)

        # 소프트맥스 함수는 마지막 차원인 key의 문장 길이 방향으로 수행된다.
        # attention weight : (batch_size, num_heads, query의 문장 길이, key의 문장 길이)
        attention_weights = self.softmax(logits)

        # output : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        output = torch.matmul(attention_weights, value)

        return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.scaled_dot_product_attention = scaled_dot_product_attention()

        assert d_model % self.num_heads == 0

        # d_model을 num_heads로 나눈 값.
        # 논문 기준 : 64
        self.depth = d_model // self.num_heads

        # WQ, WK, WV에 해당하는 밀집층 정의
        self.query_dense = nn.Linear(d_model, d_model)
        self.key_dense = nn.Linear(d_model, d_model)
        self.value_dense = nn.Linear(d_model, d_model)

        # WO에 해당하는 밀집층 정의
        self.dense = nn.Linear(d_model, d_model)

    # num_heads 개수만큼 q, k, v를 split하는 함수
    def split_heads(self, inputs, batch_size):
        inputs = inputs.reshape(batch_size, -1, self.num_heads, self.depth)
        return torch.permute(inputs, (0, 2, 1, 3))

    def forward(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = query.shape[0]

        # 1. WQ, WK, WV에 해당하는 밀집층 지나기
        # q : (batch_size, query의 문장 길이, d_model)
        # k : (batch_size, key의 문장 길이, d_model)
        # v : (batch_size, value의 문장 길이, d_model)
        # 참고) 인코더(k, v)-디코더(q) 어텐션에서는 query 길이와 key, value의 길이는 다를 수 있다.
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # 2. 헤드 나누기
        # q : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        # k : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
        # v : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # 3. 스케일드 닷 프로덕트 어텐션. 앞서 구현한 함수 사용.
        # (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        scaled_attention, _ = self.scaled_dot_product_attention(query, key, value, mask)
        # (batch_size, query의 문장 길이, num_heads, d_model/num_heads)
        scaled_attention = torch.permute(scaled_attention, (0, 2, 1, 3))

        # 4. 헤드 연결(concatenate)하기
        # (batch_size, query의 문장 길이, d_model)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)

        # 5. WO에 해당하는 밀집층 지나기
        # (batch_size, query의 문장 길이, d_model)
        outputs = self.dense(concat_attention)

        return outputs

#padding mask input : (batch_size, key의 문장 길이)
def create_padding_mask(x):
    c = torch.zeros_like(x)
    mask = (x == c).float()
    # (batch_size, 1, 1, key의 문장 길이)
    return mask.reshape(mask.shape[0], 1, 1, mask.shape[1])

class encoder_layer(nn.Module):
    def __init__(self, emb_dim, hid_dim, num_heads, dropout):
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.attention = MultiHeadAttention(
            self.emb_dim, self.num_heads, name="attention")

        self.dense1 = nn.Linear(emb_dim, hid_dim)
        self.dense2 = nn.Linear(hid_dim, emb_dim)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.layer_dropout = nn.Dropout(dropout)

    def forward(self, inputs, padding_mask):
        # 멀티-헤드 어텐션 (첫번째 서브층 / 셀프 어텐션)
        attention = self.attention({
                'query': inputs, 'key': inputs, 'value': inputs, # Q = K = V
                'mask': padding_mask # 패딩 마스크 사용
        })
        attention = attention

        # 드롭아웃 + 잔차 연결과 층 정규화
        attention = self.layer_dropout(attention)
        attention = self.layer_norm(attention + inputs)
        
        # 포지션 와이즈 피드 포워드 신경망 (두번째 서브층)
        outputs = self.dense1(attention)
        outputs = self.relu(outputs)
        outputs = self.dense2(outputs)

        # 드롭아웃 + 잔차 연결과 층 정규화
        outputs = self.layer_dropout(outputs)
        outputs = self.layer_norm(outputs + attention)
        return outputs


class Encoder(nn.Module):
    #input dim = vocab size, emb_dim=d_model, hid_dim = dff
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, num_heads, dropout, device):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device

        self.layer_stack = nn.ModuleList([
            encoder_layer(self.emb_dim, self.hid_dim, self.num_heads, self.dropout)
            for _ in range(n_layers)])

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, inputs, padding_mask):
        # Embedding + Positional Encoding
        embedding = self.embedding(inputs)
        embedding *= torch.sqrt(torch.Tensor([self.emb_dim])).cuda()
        embedding = PositionalEncoding(embedding.shape[1], embedding.shape[2])(embedding)
        outputs = self.dropout_layer(embedding)

        for enc_layer in self.layer_stack:
            outputs = enc_layer(outputs, padding_mask)
            
        return outputs

# 디코더의 첫번째 서브층(sublayer)에서 미래 토큰을 Mask하는 함수
# output : (batch_size, 1, 1, query의 문장 길이)
def create_look_ahead_mask(x):
    mask = torch.triu(torch.ones(x.shape[1],x.shape[1]), diagonal=1)
    padding_mask = create_padding_mask(x) # 패딩 마스크도 포함
    return torch.maximum(mask, padding_mask)

# # 디코더의 첫번째 서브층(sublayer)에서 미래 토큰을 Mask하는 함수
# # output : (batch_size, 1, 1, query의 문장 길이)
# def create_look_ahead_mask(x):
#     mask = torch.triu(torch.ones_like(x), diagonal=1)
#     padding_mask = create_padding_mask(x) # 패딩 마스크도 포함
#     look_ahead_mask = mask.reshape(mask.shape[0], 1, 1, mask.shape[1])
#     return torch.maximum(look_ahead_mask, padding_mask)

class decoder_layer(nn.Module):
    def __init__(self, emb_dim, hid_dim, num_heads, dropout):
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.attention1 = MultiHeadAttention(self.emb_dim, self.num_heads, name="attention1")
        self.attention2 = MultiHeadAttention(self.emb_dim, self.num_heads, name="attention_2")

        self.dense1 = nn.Linear(emb_dim, hid_dim)
        self.dense2 = nn.Linear(hid_dim, emb_dim)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.layer_dropout = nn.Dropout(dropout)

    def forward(self, inputs, enc_outputs, padding_mask, look_ahead_mask):
        # 멀티-헤드 어텐션 (첫번째 서브층 / 마스크드 셀프 어텐션)
        attention1 = self.attention1({
                'query': inputs, 'key': inputs, 'value': inputs, # Q = K = V
                'mask': look_ahead_mask # 패딩 마스크 사용
        })

        # 잔차 연결과 층 정규화
        attention1 = self.layer_norm(attention1 + inputs)

        # 멀티-헤드 어텐션 (두번째 서브층 / 디코더-인코더 어텐션)
        attention2 = self.attention2(inputs={
            'query': attention1, 'key': enc_outputs, 'value': enc_outputs, # Q != K = V
            'mask': padding_mask # 패딩 마스크
        })

        # 드롭아웃 + 잔차 연결과 층 정규화
        attention2 = self.layer_dropout(attention2)
        attention2 = self.layer_norm(attention2 + attention1)

        # 포지션 와이즈 피드 포워드 신경망 (세번째 서브층)
        outputs = self.dense1(attention2)
        outputs = self.relu(outputs)
        outputs = self.dense2(outputs)

        # 드롭아웃 + 잔차 연결과 층 정규화
        outputs = self.layer_dropout(outputs)
        outputs = self.layer_norm(outputs + attention2)
        return outputs

class Decoder(nn.Module):
    #input dim = vocab size, emb_dim=d_model, hid_dim = dff
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, num_heads, dropout, device):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device

        self.layer_stack = nn.ModuleList([
            decoder_layer(self.emb_dim, self.hid_dim, self.num_heads, 
            self.dropout) for _ in range(n_layers)])

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, inputs, enc_outputs, padding_mask, lookahead_mask):
        # Embedding + Positional Encoding
        embedding = self.embedding(inputs)
        embedding *= torch.sqrt(torch.Tensor([self.emb_dim])).cuda()
        embedding = PositionalEncoding(embedding.shape[1], embedding.shape[2])(embedding)
        outputs = self.dropout_layer(embedding)

        #def decoder_layer(emb_dim, hid_dim, num_heads, dropout, inputs, enc_outputs, padding_mask, look_ahead_mask)
        for dec_layer in self.layer_stack:
            outputs = dec_layer(outputs, enc_outputs, padding_mask, lookahead_mask)

        return outputs

class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.fc_out = nn.Linear(decoder.emb_dim, decoder.input_dim)

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg):
        #encoder
        enc_padding_mask = create_padding_mask(src)
        #decoder
        dec_padding_mask = create_padding_mask(src)
        look_ahead_mask = create_look_ahead_mask(trg)

        enc_outputs = self.encoder(src, enc_padding_mask)
        outputs = self.decoder(trg, enc_outputs, dec_padding_mask, look_ahead_mask)
        outputs = self.fc_out(outputs)

        #outputs size : (batch_num, seq_len, target_vocab_size)
        return outputs

#warmup_steps 보다 작으면 선형적으로 증가 이후 역제곱근으로 감소
class CustomSchedule(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        def lr_lambda(step):
            arg1 = max(step,1) ** (-0.5)
            arg2 = max(step,1) * (warmup_steps**-1.5)
            return d_model ** (-0.5) * min(arg1, arg2)

        super(CustomSchedule, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)

if __name__ == "__main__":
    # sample_pos_encoding = PositionalEncoding(50, 128)
    # a=torch.randn(5,50,128)
    # print(sample_pos_encoding(a).shape)
    # plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
    # plt.xlabel('Depth')
    # plt.xlim((0, 128))
    # plt.ylabel('Position')
    # plt.colorbar()
    # plt.show()

    # np.set_printoptions(suppress=True)
    # temp_k = torch.Tensor([[10,0,0],
    #                     [0,10,0],
    #                     [0,0,10],
    #                     [0,0,10]])  # (4, 3)

    # temp_v = torch.Tensor([[  1,0],
    #                     [  10,0],
    #                     [ 100,5],
    #                     [1000,6]])  # (4, 2)
    # temp_q = torch.Tensor([[0, 10, 0]])  # (1, 3)

    # temp_out, temp_attn = scaled_dot_product_attention()(temp_q, temp_k, temp_v, torch.Tensor([[1.0, 0.0, 1.0, 0.0]]))
    # print(temp_attn) # 어텐션 분포(어텐션 가중치의 나열)
    # print(temp_out) # 어텐션 값
    # print(temp_out.type())

    # a = torch.Tensor([[1, 21, 777, 0, 0]])
    # print(a.shape)
    # print(create_padding_mask(a))
    # print(create_padding_mask(a).shape)
    # print(create_look_ahead_mask(a).shape)

    # class modela(nn.Module):
    #     def __init__(self):
    #         super(modela, self).__init__()
    #         self.linear = nn.Linear(10, 10)
    #         self.activation = nn.ReLU()
    #     def forward(self, x):
    #         return self.activation(self.linear1(x))

    # model=modela()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    # scheduler = CustomSchedule(optimizer, d_model = 128)
    # print(optimizer.param_groups[0]['lr'])
    # a=[]
    # for step in range(200000):
    #     a.append(optimizer.param_groups[0]['lr'])
    #     scheduler.step()
    # step = range(200000)
    # plt.plot(step,a)
    # plt.show()

    # print(torch.Tensor(encoder_index).shape)
    # print(torch.Tensor(decoder_input_index).shape)

    # print(encoder_index[100])
    # print(decoder_input_index[100])

    # print(create_look_ahead_mask(torch.randn(4,4)))
    print(encoder_index[1])
    print(decoder_input_index[100])
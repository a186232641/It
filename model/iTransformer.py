import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer,ProbAttention
from layers.Embed import DataEmbedding_inverted
import numpy as np
from RevIN import RevIN

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.pyramid = configs.pyramid
        self.use_RevIN = configs.use_RevIN
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        DataEmedding_blocks = [DataEmbedding_inverted(self.seq_len // (2 ** (self.pyramid)),configs.d_model,configs.embed, configs.freq,configs.dropout)]+\
            [DataEmbedding_inverted(self.seq_len // (2 ** (self.pyramid-i)),configs.d_model,configs.embed, configs.freq,configs.dropout) for i in range(self.pyramid + 1)]
        self.DataEmedding_blocks = nn.ModuleList(DataEmedding_blocks)
        self.embedding = [DataEmbedding_inverted]
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.use_RevIN:
            self.revin = RevIN(configs.enc_in)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec,embeding_block=None):

        #x_enc -->去除时间变量后的  x_mark_enc 时间变量
        # Normalization from Non-stationary Transformer
        self.forecast_embed = embeding_block
        if self.use_RevIN:
            x_enc = self.revin(x_enc, 'norm')
        else:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)#
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)([128, 48, 6])--->torch.Size([128, 10, 128])时间嵌入，
        # enc_out = self.forecast_embed(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens#
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # enc --》torch.Size([128, 10, 128])
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        

        # B N E -> B N S -> B S N #torch.Size([128, 10, 128]) --> torch.Size([128, 12, 6])
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] #全连接然后改变位置 然后只选择 N 原来所有的变量
        # torch.Size([128, 12, 6])
        # filter the covariates 
        # dec_out  
        # torch.Size([128, 12, 6])
        # # De-Normalization from Non-stationary Transformer
       
        if self.use_RevIN:
            dec_out = self.revin(dec_out, 'denorm')
        else:
            #
             dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))#torch.Size([128, 12, 6])
             dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))#torch.Size([128, 12, 6])

        return dec_out#torch.Size([128, 12, 6])


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        enc_input = x_enc[:, :self.seq_len, :]#torch.Size([128, 48, 6])
        # enc_mark_input = x_mark_enc[:, :self.seq_len, :]#
        # x_enc_list = [x_enc[:, -self.seq_len // (2 ** self.pyramid):, :]]#切割选取子张量
        # x_mark_enc_list = [x_mark_enc[:, -self.seq_len // (2 ** self.pyramid):, :]]

        # for i in range(self.pyramid):
        #     x_enc_list.append(enc_input[:, -self.seq_len // (2 ** (self.pyramid - i - 1)):
        #                                        -self.seq_len // (2 ** (self.pyramid - i)), :])
        #     x_mark_enc_list.append(enc_mark_input[:, -self.seq_len // (2 ** (self.pyramid - i - 1)):
        #                                        -self.seq_len // (2 ** (self.pyramid - i)), :])
        # dec_out = 0
        # num_out = 0

        # for current_x_enc,cunrrent_x_enc_mark,Data_emd in zip(x_enc_list,x_mark_enc_list,self.DataEmedding_blocks):
        #     dec_out += self.forecast(current_x_enc,cunrrent_x_enc_mark,x_dec,x_mark_dec,Data_emd)
        #     num_out += 1
        # return dec_out/num_out
        



        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

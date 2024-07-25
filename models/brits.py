import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import utils
import argparse
import data_loader

from . import rits
from sklearn import metrics

from ipdb import set_trace

SEQ_LEN = 48
RNN_HID_SIZE = 64

class TransformerBlock(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, hidden_dim):
        super(TransformerBlock, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, src):
        return self.transformer_encoder(src)

class Model(nn.Module):
    def __init__(self, rnn_hid_size, impute_weight, label_weight, num_heads=8, num_layers=6, hidden_dim=256):
        super(Model, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.label_weight = label_weight
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.build()

    def build(self):
        self.rits_f = rits.Model(self.rnn_hid_size, self.impute_weight, self.label_weight)
        self.rits_b = rits.Model(self.rnn_hid_size, self.impute_weight, self.label_weight)
        self.transformer_block = TransformerBlock(input_size=self.rnn_hid_size, num_heads=self.num_heads, num_layers=self.num_layers, hidden_dim=self.hidden_dim)

    def forward(self, data):
        ret_f = self.rits_f(data, 'forward')
        ret_b = self.reverse(self.rits_b(data, 'backward'))

        # Apply transformer block to the imputations
        ret_f['imputations'] = self.transformer_block(ret_f['imputations'].permute(1, 0, 2)).permute(1, 0, 2)
        ret_b['imputations'] = self.transformer_block(ret_b['imputations'].permute(1, 0, 2)).permute(1, 0, 2)

        ret = self.merge_ret(ret_f, ret_b)

        return ret

    def merge_ret(self, ret_f, ret_b):
        loss_f = ret_f['loss']
        loss_b = ret_b['loss']
        loss_c = self.get_consistency_loss(ret_f['imputations'], ret_b['imputations'])

        loss = loss_f + loss_b + loss_c

        predictions = (ret_f['predictions'] + ret_b['predictions']) / 2
        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2

        ret_f['loss'] = loss
        ret_f['predictions'] = predictions
        ret_f['imputations'] = imputations

        return ret_f

    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = list(range(tensor_.size()[1]))[::-1]
            indices = Variable(torch.LongTensor(indices), requires_grad=False)

            if torch.cuda.is_available():
                indices = indices.cuda()

            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret

    def run_on_batch(self, data, optimizer, epoch=None):
        ret = self(data)

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret

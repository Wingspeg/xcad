import random

from torch.nn import functional as F
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
import os
path_dir = os.getcwd()

class ConvTransR(torch.nn.Module):
    def __init__(self, num_relations, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3, use_bias=True):
        super(ConvTransR, self).__init__()
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_relations*2)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(embedding_dim)
        # self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(embedding_dim)

    def forward(self, embedding, emb_rel, triplets, nodes_id=None, mode="train", negative_rate=0):

        e1_embedded_all = F.tanh(embedding)
        batch_size = len(triplets)
        # if mode=="train":
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        e2_embedded = e1_embedded_all[triplets[:, 2]].unsqueeze(1)
        # else:
        #     e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        #     e2_embedded = e1_embedded_all[triplets[:, 2]].unsqueeze(1)
        stacked_inputs = torch.cat([e1_embedded, e2_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, emb_rel.transpose(1, 0))
        return x


class ConvTransE(torch.nn.Module):
    def __init__(self, num_entities, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3, use_bias=True,
                 use_decoder_feat=False, feat_dim_list=None, feat_alpha_fixed=None,
                 use_compat=False):

        super(ConvTransE, self).__init__()
        # 初始化relation embeddings
        # self.emb_rel = torch.nn.Embedding(num_relations, embedding_dim, padding_idx=0)

        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(embedding_dim)
        # self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(embedding_dim)
        self.use_decoder_feat = use_decoder_feat
        self.use_compat = use_compat

        if self.use_decoder_feat:
            assert feat_dim_list is not None, "feat_dim_list required when use_decoder_feat=True"
            algo_dim, data_dim, compute_dim = feat_dim_list
            # 三段独立投影层
            self.proj_algo    = nn.Linear(algo_dim, embedding_dim)
            self.proj_data    = nn.Linear(data_dim, embedding_dim)
            self.proj_compute = nn.Linear(compute_dim, embedding_dim)
            nn.init.xavier_normal_(self.proj_algo.weight)
            nn.init.xavier_normal_(self.proj_data.weight)
            nn.init.xavier_normal_(self.proj_compute.weight)
            # GPU type 段 6 个可学习 embedding
            self.gpu_emb = nn.Parameter(torch.Tensor(6, embedding_dim))
            nn.init.normal_(self.gpu_emb)
            # decoder_lambda
            if feat_alpha_fixed is not None:
                self.register_buffer("decoder_lambda", torch.tensor(float(feat_alpha_fixed)))
            else:
                self.decoder_lambda = nn.Parameter(torch.tensor(0.1))



    def forward(self, embedding, emb_rel, triplets, nodes_id=None, mode="train", negative_rate=0, partial_embeding=None):
        e1_embedded_all = F.tanh(embedding)
        batch_size = len(triplets)
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        if partial_embeding is None:
            base_score = torch.mm(x, e1_embedded_all.transpose(1, 0))
        else:
            base_score = torch.mm(x, partial_embeding.transpose(1, 0))

        if self.use_decoder_feat:
            # feat_emb: [num_ents, embedding_dim], 预拼,forward 内现算
            feat_emb = torch.cat([
                self.proj_algo(self.feat_algo),
                self.proj_data(self.feat_data),
                self.proj_compute(self.feat_compute),
                self.gpu_emb,
            ], dim=0)  # [num_ents, embedding_dim]
            feat_bias = torch.mm(x, feat_emb.transpose(1, 0))  # [batch, num_ents]
            x = base_score + self.decoder_lambda * feat_bias
        else:
            x = base_score
        if self.use_compat:
            # 兼容性头: 只对 r1_suits 正向边(rel=1), 尺度对齐后注入
            r1_mask = (triplets[:, 1] == 1)
            if r1_mask.any():
                r1_idx = r1_mask.nonzero(as_tuple=True)[0]
                h_ids = triplets[r1_idx, 0]
                feat = self.compat_feat_algo.to(x.device)[h_ids]
                compat = self.compat_head(feat)  # [n_r1, 6], 对应 entity 0-5
                # 主路在 GPU 列(0-5)的尺度
                main_gpu = x[r1_idx, 0:6]  # [n_r1, 6]
                main_std = main_gpu.std(dim=1, keepdim=True).clamp(min=1e-8)
                # compat 标准化到主路尺度
                compat_n = (compat - compat.mean(dim=1, keepdim=True)) / compat.std(dim=1, keepdim=True).clamp(min=1e-8)
                compat_scaled = compat_n * main_std  # 尺度对齐
                bias = torch.zeros_like(x)
                bias[r1_idx, 0:6] = self.compat_lambda * compat_scaled  # 列 0-5 全加(含 CPU, 但 r1 真值不会是 0, 无害)
                x = x + bias  # 非 in-place, 梯度通畅
        return x

    def forward_slow(self, embedding, emb_rel, triplets):

        e1_embedded_all = F.tanh(embedding)
        # e1_embedded_all = embedding
        batch_size = len(triplets)
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        # translate to sub space
        # e1_embedded = torch.matmul(e1_embedded, sub_trans)
        rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        e2_embedded = e1_embedded_all[triplets[:, 2]]
        score = torch.sum(torch.mul(x, e2_embedded), dim=1)
        pred = score
        return pred
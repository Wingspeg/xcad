import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from rgcn.utils import sort_and_rank, filter_score
from src.model import BaseRGCN
from src.decoder import ConvTransE

class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError

    def forward(self, g, init_ent_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            for i, layer in enumerate(self.layers):
                layer(g, [])
            return g.ndata.pop('h')
        else:
            raise NotImplementedError

class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu = 0):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.gpu = gpu

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)

        self.loss_e = torch.nn.CrossEntropyLoss()


        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda)

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)    
        
      
        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout, sequence_len=self.sequence_len)
        else:
            raise NotImplementedError 


    def forward(self, g_list, use_cuda):
        evolve_embs = []
        self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb
        for i, g in enumerate(g_list):
            g = g.to(self.gpu)
            current_h = self.rgcn.forward(g, self.h)
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            time_weight = F.sigmoid(torch.mm(self.h, self.time_gate_weight) + self.time_gate_bias)
            self.h = time_weight * current_h + (1-time_weight) * self.h
            self.h = F.normalize(self.h)
            evolve_embs.append(self.h)
        return evolve_embs, self.emb_rel

    def predict(self, test_graph, test_triplets, use_cuda, all_ans_ent=None, eval_bs=1000):
        """
        分批算 score + 分批算 rank，不返回完整 score 矩阵。
        峰值显存 ≈ [eval_bs, num_ents] ≈ 500MB (bs=1000, num_ents=124891, FP32)。

        返回: (rank_raw, rank_filter)
          - rank_*: 1-indexed rank 张量, shape=[N]
          - 与原 utils.get_total_rank 返回的 rank 完全等价
        """
        with torch.no_grad():
            # evolve_embeddings 只在分块循环外算一次（整图前向，和 triples 分批无关）
            evolve_embeddings = []
            for idx in range(len(test_graph)):
                evolve_embs, r_emb = self.forward(test_graph[idx:], use_cuda)
                evolve_embeddings.append(evolve_embs[-1])
            evolve_embeddings.reverse()

            N = len(test_triplets)
            rank_raw_list = []
            rank_filter_list = []

            for batch_start in range(0, N, eval_bs):
                batch_end = min(N, batch_start + eval_bs)
                triples_batch = test_triplets[batch_start:batch_end]

                # === entity score: [batch, num_ents] ===
                score_list = self.decoder_ob.forward(evolve_embeddings, r_emb, triples_batch, mode="test")
                # 聚合：softmax(dim=1) 在 num_ents 维，sum(dim=-1) 在 T 维
                score_list = [_.unsqueeze(2) for _ in score_list]
                scores_batch = torch.cat(score_list, dim=2)
                scores_batch = torch.softmax(scores_batch, dim=1)
                scores_batch = torch.sum(scores_batch, dim=-1)  # [batch, num_ents]
                target_e = triples_batch[:, 2]

                # 1) raw rank
                rank_raw_list.append(sort_and_rank(scores_batch, target_e))

                # 2) filter rank: 原地把已知正确答案的 score 设为 -1e7
                #    all_ans_ent[(h, r)] 的 key 只与本 batch 的 triples 有关，
                #    不会跨 batch 漏过滤或错位
                if all_ans_ent is not None:
                    filter_score(triples_batch, scores_batch, all_ans_ent)
                    rank_filter_list.append(sort_and_rank(scores_batch, target_e))

                del scores_batch, score_list

            # 拼成完整 rank，+1 转 1-indexed（与原 get_total_rank 完全一致）
            rank_raw = torch.cat(rank_raw_list) + 1
            rank_filter = torch.cat(rank_filter_list) + 1
            return rank_raw, rank_filter

    def get_ft_loss(self, glist, triple_list,  use_cuda):
        #"""
        #:param glist:
        #:param triplets:
        #:param use_cuda:
        #:return:
        #"""
        glist = [g.to(self.gpu) for g in glist]
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        # for step, triples in enumerate(triple_list):
        evolve_embeddings = []
        for idx in range(len(glist)):
            evolve_embs, r_emb = self.forward(glist[idx:], use_cuda)
            evolve_embeddings.append(evolve_embs[-1])
        evolve_embeddings.reverse()
        scores_ob = self.decoder_ob.forward(evolve_embeddings, r_emb, triple_list[-1])#.view(-1, self.num_ents)
        for idx in range(len(glist)):
            loss_ent += self.loss_e(scores_ob[idx], triple_list[-1][:, 2])
        return loss_ent

    def get_loss(self, glist, triples, prev_model, use_cuda):
        """
        :param glist:
        :param triplets:
        :param use_cuda:
        :return:
        """
        ent_eval_bs = 2000
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        evolve_embeddings = []
        for idx in range(len(glist)):
            evolve_embs, r_emb = self.forward(glist[idx:], use_cuda)
            evolve_embeddings.append(evolve_embs[-1])
        evolve_embeddings.reverse()
        if self.entity_prediction:
            N = triples.shape[0]
            num_steps = len(glist)
            # 分批：对 triples 按 ent_eval_bs 分块，每块独立调 decoder + 算 CE
            for batch_start in range(0, N, ent_eval_bs):
                batch_end = min(N, batch_start + ent_eval_bs)
                triples_batch = triples[batch_start:batch_end]
                # decoder 只在该 batch 上做 mm，返回 len(glist) 个 [batch_size, num_ents]
                scores_batch = self.decoder_ob.forward(evolve_embeddings, r_emb, triples_batch)
                for step_idx in range(num_steps):
                    # CrossEntropyLoss(reduction='mean') 对该 batch 求 mean CE
                    loss_ent += self.loss_e(scores_batch[step_idx], triples_batch[:, 2])
                # scores_batch 出了内层循环即超出作用域，PyTorch 引用计数回收显存
            # 归一化：保持 loss_ent = Σ_step mean_CE 的原版语义，等价于 (1/N) * Σ_step Σ_i CE(step,i)
            loss_ent = loss_ent / N
        return loss_ent


import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from src.model import BaseRGCN
from src.decoder import ConvTransE, ConvTransR
from rgcn.utils import sort_and_rank, filter_score, filter_score_r


class RGCNCell(BaseRGCN):
    def __init__(self, *args, use_hetero=False, n_msg_basis=2, use_rel_decay=False, rel_decay=None, **kwargs):
        super(RGCNCell, self).__init__(*args, use_hetero=use_hetero, n_msg_basis=n_msg_basis, use_rel_decay=use_rel_decay, rel_decay=rel_decay, **kwargs)

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
                                  activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb,
                                  use_hetero=self.use_hetero, n_msg_basis=self.n_msg_basis,
                                  use_rel_decay=self.use_rel_decay, rel_decay=self.rel_decay)
        else:
            raise NotImplementedError

    def forward(self, g, init_ent_emb, init_rel_emb, step=None):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i], step=step)
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h, step=step)
            else:
                for layer in self.layers:
                    layer(g, [], step=step)
            return g.ndata.pop('h')


class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu=0, analysis=False,
                 use_hetero=False, n_msg_basis=2,
                 use_node_feat=False, feat_dir=None, feat_fusion='residual', feat_scope='all', feat_alpha_fixed=None,
                 use_decoder_feat=False, decoder_feat_lambda=None,
                 use_rel_decay=False,
                 use_compat=False, compat_lambda=None, compat_aux_weight=0.5):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu
        self.use_hetero = use_hetero
        self.n_msg_basis = n_msg_basis
        self.use_node_feat = use_node_feat
        self.feat_dir = feat_dir
        self.feat_fusion = feat_fusion
        self.feat_scope = feat_scope
        self.feat_alpha_fixed = feat_alpha_fixed
        self.use_decoder_feat = use_decoder_feat
        self.decoder_feat_lambda = decoder_feat_lambda
        self.use_rel_decay = use_rel_decay
        self.use_compat = use_compat
        self.compat_lambda_val = compat_lambda
        self.compat_aux_weight = compat_aux_weight
        self._aux_loss_step = 0

        # 节点段边界(全局 id 空间, 与 export_to_regcn_format.py 一致)
        self.N_ALGO    = 102610
        self.N_DATA    = 20378
        self.N_COMPUTE = 1897
        self.N_GPU     = 6

        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        # baseline: 随机 normal_ 初始化 (use_node_feat 时也被保留,融合用)
        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)


        # ── 关系感知时间衰减 ──
        if self.use_rel_decay:
            self.rel_decay = nn.Parameter(torch.zeros(num_rels * 2))
                # ── 兼容性头 (algorithm feature → GPU compatibility scores) ──
        if self.use_compat:
            self.compat_feat_algo = np.load(os.path.join(feat_dir, "node_feat_algo.npy"))
            # register as buffer so .cuda() works
            self.register_buffer("_compat_feat_algo", torch.from_numpy(self.compat_feat_algo).float())
            # don't keep the numpy copy
            del self.compat_feat_algo
            self.compat_head = nn.Sequential(
                nn.Linear(13, 64),
                nn.ReLU(),
                nn.Linear(64, 6),
            )
            nn.init.xavier_normal_(self.compat_head[0].weight)
            nn.init.xavier_normal_(self.compat_head[2].weight)
            if self.compat_lambda_val is not None:
                self.register_buffer("_compat_lambda", torch.tensor(float(self.compat_lambda_val)))
            else:
                self._compat_lambda = nn.Parameter(torch.tensor(0.1))
# ── node-feat 特征注入 ──
        if self.use_node_feat or self.use_decoder_feat:
            # 加载特征矩阵,register_buffer(不训练,跟随 .cuda())
            feat_algo    = np.load(os.path.join(feat_dir, "node_feat_algo.npy"))
            feat_data    = np.load(os.path.join(feat_dir, "node_feat_data.npy"))
            feat_compute = np.load(os.path.join(feat_dir, "node_feat_compute.npy"))
            self.register_buffer("feat_algo",    torch.from_numpy(feat_algo).float())
            self.register_buffer("feat_data",    torch.from_numpy(feat_data).float())
            self.register_buffer("feat_compute", torch.from_numpy(feat_compute).float())
            assert feat_algo.shape[0]   == self.N_ALGO
            assert feat_data.shape[0]   == self.N_DATA
            assert feat_compute.shape[0] == self.N_COMPUTE

            assert feat_fusion in ('replace', 'residual')
            assert feat_scope  in ('all', 'algo')

            # 三段独立投影层,xavier 初始化
            self.proj_algo    = nn.Linear(feat_algo.shape[1],    h_dim)
            self.proj_data    = nn.Linear(feat_data.shape[1],    h_dim)
            self.proj_compute = nn.Linear(feat_compute.shape[1], h_dim)
            nn.init.xavier_normal_(self.proj_algo.weight)
            nn.init.xavier_normal_(self.proj_data.weight)
            nn.init.xavier_normal_(self.proj_compute.weight)

            # GPU type 段 6 个可学习 embedding,normal_ 初始化
            self.gpu_emb = nn.Parameter(torch.Tensor(self.N_GPU, h_dim))
            nn.init.normal_(self.gpu_emb)

            # feat_alpha: 可学习或固定 buffer
            if feat_alpha_fixed is not None:
                self.register_buffer("feat_alpha", torch.tensor(float(feat_alpha_fixed)))
            else:
                self.feat_alpha = nn.Parameter(torch.tensor(0.1))

        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
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
                             use_cuda,
                             analysis,
                             use_hetero=self.use_hetero,
                             n_msg_basis=self.n_msg_basis,
                             use_rel_decay=self.use_rel_decay,
                             rel_decay=self.rel_decay if self.use_rel_decay else None)

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)

        self.relation_cell_1 = nn.GRUCell(self.h_dim*2, self.h_dim)

        if decoder_name == "convtranse":
            feat_dim_list = None
            if self.use_node_feat or self.use_decoder_feat:
                feat_dim_list = [
                    self.feat_algo.shape[1],
                    self.feat_data.shape[1],
                    self.feat_compute.shape[1],
                ]
            # decoder-feat 依赖 node-feat 的特征矩阵;lambda 由 decoder_feat_lambda 独立控制
            use_decoder_feat = self.use_decoder_feat
            self.decoder_ob = ConvTransE(
                num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout,
                use_decoder_feat=use_decoder_feat,
                use_compat=self.use_compat,
                feat_dim_list=feat_dim_list,
                feat_alpha_fixed=self.decoder_feat_lambda,
            )
            self.rdecoder = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError

    # ============================================================
    def compute_dynamic_emb(self):
        if not self.use_node_feat:
            return self.dynamic_emb  # bit 级同原版

        p_algo    = self.proj_algo(self.feat_algo)
        p_data    = self.proj_data(self.feat_data)
        p_compute = self.proj_compute(self.feat_compute)
        feat_proj = torch.cat([p_algo, p_data, p_compute, self.gpu_emb], dim=0)  # [num_ents, h_dim]

        if self.feat_fusion == 'replace':
            return feat_proj
        else:  # residual
            if self.feat_scope == 'all':
                return self.dynamic_emb + self.feat_alpha * feat_proj
            else:  # 'algo': 只对 Algorithm 段做残差融合
                alpha_p  = self.feat_alpha * self.proj_algo(self.feat_algo)
                algo_part = self.dynamic_emb[:self.N_ALGO] + alpha_p
                rest     = self.dynamic_emb[self.N_ALGO:]
                return torch.cat([algo_part, rest], dim=0)

    def forward(self, g_list, static_graph, use_cuda):
        gate_list = []
        degree_list = []

        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            emb = self.compute_dynamic_emb()
            static_graph.ndata['h'] = torch.cat((emb, self.words_emb), dim=0)
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            emb = self.compute_dynamic_emb()
            self.h = F.normalize(emb) if self.layer_norm else emb[:, :]
            # 把 node-feat buffers 传给 decoder_ob,让 decoder 自己做投影偏置
            if self.use_decoder_feat:
                self.decoder_ob.feat_algo    = self.feat_algo
                self.decoder_ob.feat_data    = self.feat_data
                self.decoder_ob.feat_compute = self.feat_compute
            if self.use_compat:
                self.decoder_ob.compat_head       = self.compat_head
                self.decoder_ob.compat_feat_algo  = self._compat_feat_algo
                self.decoder_ob.compat_lambda     = self._compat_lambda
            static_emb = None

        history_embs = []

        for i, g in enumerate(g_list):
            step = len(g_list) - 1 - i
            g = g.to(self.gpu)
            temp_e = self.h[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_rels * 2, self.h_dim).float()
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1], :]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean
            if i == 0:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            else:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.h_0)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0], step=step)
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            time_weight = F.sigmoid(torch.mm(self.h, self.time_gate_weight) + self.time_gate_bias)
            self.h = time_weight * current_h + (1-time_weight) * self.h
            history_embs.append(self.h)
        return history_embs, static_emb, self.h_0, gate_list, degree_list

    def predict(self, test_graph, num_rels, static_graph, test_triplets, use_cuda,
                all_ans_ent=None, all_ans_rel=None, eval_bs=1000):
        """
        分批算 score + rank, 避免 OOM.
        """
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels
            all_triples = torch.cat((test_triplets, inverse_test_triplets))
            all_triples = all_triples.to(self.gpu)

            evolve_embs, _, r_emb, _, _ = self.forward(test_graph, static_graph, use_cuda)
            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

            N = all_triples.size(0)
            rank_raw_list, rank_filter_list = [], []
            rank_raw_list_r, rank_filter_list_r = [], []

            for start in range(0, N, eval_bs):
                end = min(start + eval_bs, N)
                batch_triples = all_triples[start:end]

                score = self.decoder_ob.forward(embedding, r_emb, batch_triples, mode="test")
                target_e = batch_triples[:, 2]

                rank_raw_list.append(sort_and_rank(score, target_e))
                if all_ans_ent is not None:
                    filter_score(batch_triples, score, all_ans_ent)
                    rank_filter_list.append(sort_and_rank(score, target_e))

                score_rel = self.rdecoder.forward(embedding, r_emb, batch_triples, mode="test")
                target_r = batch_triples[:, 1]
                rank_raw_list_r.append(sort_and_rank(score_rel, target_r))
                if all_ans_rel is not None:
                    filter_score_r(batch_triples, score_rel, all_ans_rel)
                    rank_filter_list_r.append(sort_and_rank(score_rel, target_r))

                del score, score_rel

            rank_raw = torch.cat(rank_raw_list) + 1
            rank_filter = torch.cat(rank_filter_list) + 1
            rank_raw_r = torch.cat(rank_raw_list_r) + 1
            rank_filter_r = torch.cat(rank_filter_list_r) + 1

            return all_triples, rank_raw, rank_filter, rank_raw_r, rank_filter_r

    def get_loss(self, glist, triples, static_graph, use_cuda):
        loss_ent = torch.zeros(1, requires_grad=True).cuda().to(self.gpu) if use_cuda else torch.zeros(1, requires_grad=True)
        loss_rel = torch.zeros(1, requires_grad=True).cuda().to(self.gpu) if use_cuda else torch.zeros(1, requires_grad=True)
        loss_static = torch.zeros(1, requires_grad=True).cuda().to(self.gpu) if use_cuda else torch.zeros(1, requires_grad=True)

        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu)

        evolve_embs, static_emb, r_emb, _, _ = self.forward(glist, static_graph, use_cuda)
        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

        if self.entity_prediction:
            ent_eval_bs = 2000
            num_rows = all_triples.size(0)
            loss_ent_sum = torch.zeros((), device=all_triples.device, dtype=torch.float32)
            for start in range(0, num_rows, ent_eval_bs):
                end = min(start + ent_eval_bs, num_rows)
                batch_triples = all_triples[start:end]
                batch_scores = self.decoder_ob.forward(pre_emb, r_emb, batch_triples).view(-1, self.num_ents)
                loss_ent_sum = loss_ent_sum + F.cross_entropy(batch_scores, batch_triples[:, 2], reduction="sum")
                del batch_scores
            loss_ent += loss_ent_sum / num_rows

        # ── compat 辅助分类 loss: 直接强监督 compat_head (5-GPU 分类) ──
        if self.use_compat and self.compat_aux_weight > 0:
            r1_fwd_mask = (all_triples[:, 1] == 1)  # 只正向 r1 边
            if r1_fwd_mask.any():
                r1_fwd = all_triples[r1_fwd_mask]
                h_ids = r1_fwd[:, 0]
                gpu_labels = r1_fwd[:, 2]  # 1-5
                feat = self._compat_feat_algo[h_ids]
                compat_logits = self.compat_head(feat)  # [n_r1, 6]
                aux_loss = F.cross_entropy(compat_logits, gpu_labels)
                self._aux_loss_step += 1
                if self._aux_loss_step % 1000 == 1:
                    print(f"[epoch?]  compat_aux_loss={aux_loss.item():.6f}  (step={self._aux_loss_step})")
                loss_ent += self.compat_aux_weight * aux_loss

        if self.relation_prediction:
            score_rel = self.rdecoder.forward(pre_emb, r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)
            loss_rel += self.loss_r(score_rel, all_triples[:, 1])

        if self.use_static:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
        return loss_ent, loss_rel, loss_static

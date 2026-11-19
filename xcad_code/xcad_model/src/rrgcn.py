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


def _loss_accumulator(device):
    """
    返回一个 0 值、可反传、且【非 leaf】的损失累加器, 供 get_loss 里的 `+=` 使用。

    为什么不能直接写 torch.zeros(1, requires_grad=True, device=...):
      旧代码是 `torch.zeros(1, requires_grad=True).cuda().to(gpu)`, 那次设备拷贝会被
      autograd 记录, 结果带 grad_fn(非 leaf), 所以后面 `loss_ent += ...` 合法;
      而 CPU 上直接构造出来的就是 leaf, `+=` 会抛
      "a leaf Variable that requires grad is being used in an in-place operation"。
      这里用一次 `+ 0` 复现"非 leaf"这一前提, 数值与旧 GPU 路径完全一致。
    """
    z = torch.zeros(1, requires_grad=True, device=device)
    return z + torch.zeros(1, device=device)


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
                 use_rel_evolution=True,
                 use_compat=False, compat_lambda=None, compat_aux_weight=0.5,
                 compat_class_weights=None,
                 gpu_base=0,
                 use_base_emb=False, base_map_path=None,
                 use_base_feat=False, base_feat_path=None, base_alpha_init=1.0,
                 use_base_prior=False, base_prior_path=None, base_prior_alpha_init=1.0,
                 use_head_prior=False, head_prior_path=None, head_prior_alpha_init=1.0,
                 use_prior_gate=False,
                 use_prior_ts=False, base_prior_ts_path=None, head_prior_ts_path=None,
                 prior_lo=0, prior_n=0):
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
        # 设备解析: 仅在 use_cuda 且 gpu>=0 时用 cuda:gpu, 否则退回 CPU
        # (旧代码直接 .to(self.gpu), --gpu -1 时会抛 "Device index must not be negative")
        self.device = torch.device('cuda:%d' % gpu) if (use_cuda and gpu >= 0) else torch.device('cpu')
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
        # R1-1 具名消融①: False 时跳过 GRUCell, 每个快照用静态 emb_rel 做消息传递
        # (默认 True = 原行为, 不加 flag 时代码路径逐位不变)
        self.use_rel_evolution = use_rel_evolution
        self.use_compat = use_compat
        self.compat_lambda_val = compat_lambda
        self.compat_aux_weight = compat_aux_weight
        # 稀有 GPU 类加权(2026-11-18 用户放行): 只加在 compat 辅助 CE 上。
        # None = 逐位等价于原实现(F.cross_entropy 的 weight=None 分支)。
        # 为什么加在这里而不是主 CE: §31/§37 实测完整模型的 r1 读数 = compat 头直读
        # (无约束 H@1 = 段内 = compat 0.884843), 编码器在 r1 无可报独立贡献 => 能改变
        # 稀有类排名的训练信号只有这条辅助损失。
        self.compat_class_weights = (None if compat_class_weights is None else
                                     torch.as_tensor(compat_class_weights, dtype=torch.float32,
                                                     device=self.device))
        self._aux_loss_step = 0
        # GPU 段起始 id: 0 = 旧局部编号, num_ents-6 = 全局四段编号(由 main.py 按数据判定)
        self.gpu_base = gpu_base
        # ── base 条件化(§20 定形状, §21 定死因) ──
        # 动机: maas 上 84.8% 的测试边其 head 从未在历史出现, 它的 dynamic_emb[id] 是没学过
        # 的随机向量 => 打分在期望上退化成 tail 边际 (§18/§19: 该层模型 0.1233 < tail_prior
        # 0.1651)。同一批边上【直接查 base 的历史卡型直方图】(不学习)能到 0.2980。
        # 两条实现, 差别就是 §21.3 Q1 的读数:
        #   --base-emb  : 空白可学习表 base_emb[base_of[id]]。已跑(§21): 该层 0.1510,
        #                 但探针显示表里没编码卡型偏好(acc 0.1224 vs 随机 0.1152)。
        #   --base-feat : 把那张历史统计表当【输入特征】灌进来, 模型只学一个投影 Linear->h_dim。
        # 共同点: 只加在 head 段; base_of 里 tail 段与无父实体都是 pad 索引, 乘 mask 置 0
        #         => 30 个可打分实体的表征逐位不变, 不会被加一个公共偏置。
        self.use_base_emb = use_base_emb
        self.use_base_feat = use_base_feat
        self.use_base_prior = use_base_prior
        self.use_head_prior = use_head_prior
        self.use_prior_gate = use_prior_gate
        if self.use_base_emb or self.use_base_feat or self.use_base_prior:
            assert not use_node_feat, "--base-emb/--base-feat/--base-prior 与 --node-feat 不组合(maas 臂不用 node-feat)"
            _bo = torch.from_numpy(np.load(base_map_path)).long()
            assert _bo.shape == (num_ents,), f"base_of 长度 {_bo.shape} != num_ents {num_ents}"
            _pad = int(_bo.max())
            self.base_pad_index = _pad
            self.register_buffer("base_of", _bo)
            self.register_buffer("base_valid", (_bo < _pad).float().unsqueeze(1))
            print(f"[base] map={base_map_path}  n_bases={_pad}  "
                  f"有父实体={int(self.base_valid.sum())}/{num_ents}  pad={_pad}")

        if self.use_base_emb:
            self.base_emb = torch.nn.Parameter(torch.Tensor(_pad + 1, h_dim)).float()
            torch.nn.init.normal_(self.base_emb)   # 与 dynamic_emb 同 scale, 不引入新超参

        if self.use_base_feat:
            _bf = torch.from_numpy(np.load(base_feat_path)).float()
            assert _bf.shape[0] == _pad + 1, f"base_feat 行数 {_bf.shape[0]} != n_bases+1 {_pad+1}"
            assert abs(float(_bf[_pad].abs().max())) < 1e-6, "pad 行应为全 0(无信息父)"
            self.register_buffer("base_feat", _bf)
            self.base_proj = torch.nn.Linear(_bf.shape[1], h_dim)
            torch.nn.init.xavier_normal_(self.base_proj.weight)
            torch.nn.init.zeros_(self.base_proj.bias)
            # 可学习标量: z-score 后的特征 + xavier 投影在 init 时量级已接近 dynamic_emb,
            # 给模型一个"这一项整体该信多少"的自由度, 不需要人工调尺度。
            self.base_alpha = torch.nn.Parameter(torch.tensor(float(base_alpha_init)))
            print(f"[base-feat] {base_feat_path} shape={tuple(_bf.shape)} "
                  f"alpha_init={base_alpha_init}  proj xavier(fan={_bf.shape[1]})")

        # ── base 条件的【输出层】先验偏置(§22 指向: 换注入位置, 不换信息) ──
        # §22 Q1: 同一张历史统计表在闭式口径下 全新 head 0.2981 / 复发边 0.4448, 而灌进 head
        # 表征(--base-feat)只吃到 0.1551 / 0.2015; 且增益只落在"表本身答对"的边上 => 信息进来了,
        # 但经过"head 向量偏移 -> 与候选向量做双线性"这条路, 表里【候选之间谁比谁强】没被保留。
        # 统计表的自然形状是"每个候选一个分数", 所以直接加在候选的 logit 上:
        #   score[head -> 卡型 k] += alpha * log1p(count[base(head), k])
        # 只有 [prior_lo, prior_lo+prior_n) = 卡型段 的列被改动; head 段的列一位不动。
        # 逆向三元组(subject 是卡型实体)的 base_of = pad => 偏置恒 0 => 不污染 head 排名。
        if self.use_base_prior:
            _bp = torch.from_numpy(np.load(base_prior_path)).float()
            assert _bp.shape == (_pad + 1, prior_n), (
                f"base_prior 形状 {_bp.shape} != (n_bases+1, prior_n) = ({_pad + 1}, {prior_n})")
            assert abs(float(_bp[_pad].abs().max())) < 1e-6, "pad 行应为全 0(无信息父 = 不注入)"
            assert prior_lo >= 0 and prior_lo + prior_n <= num_ents, (
                f"偏置列段 [{prior_lo}, {prior_lo + prior_n}) 越界 num_ents={num_ents}")
            self.prior_lo, self.prior_n = int(prior_lo), int(prior_n)
            self.register_buffer("base_prior", _bp)
            # 一个可学习标量: alpha=0 退回原模型, alpha 大则接近闭式先验, 由 CE loss 自己定尺度
            self.prior_alpha = torch.nn.Parameter(torch.tensor(float(base_prior_alpha_init)))
            print(f"[base-prior] {base_prior_path} shape={tuple(_bp.shape)} "
                  f"cols=[{prior_lo},{prior_lo + prior_n})  alpha_init={base_prior_alpha_init}  "
                  f"log1p 上界={float(_bp.max()):.3f}")

        # ── E1: 第二条【输出层】先验通道 = head 实体自己的历史卡型计数 ──
        # §23 的读数把两条通道的分工量清了: base 通道管全新 head(0.2980 闭式 / 臂上 0.2877),
        # entity_memory 通道管复发边(0.7504 闭式), 而 baseprior 臂在复发层只有 0.4450 —— 因为它
        # 只带父节点信息, 子节点自己"上次落在哪张卡"一个字都没进模型。Q3 的 oracle(任一通道对)
        # = 全体 0.3947 / 复发层 0.7986, 是手工合并的上界; 这里把两张表各配一个可学习标量,
        # 让模型自己学信任度, 目标是胜过手工合并而不是复现它:
        #   score[head -> k] += alpha_b * log1p(cnt_base[base(head), k])
        #                    + alpha_m * log1p(cnt_head[head, k])
        # head 表按实体 id 直接索引, 不需要 base_of; 卡型实体自己的行强制全 0 => 逆向三元组
        # (subject 是卡型)该通道恒为 0, 与 base 通道同一条不变量。
        if self.use_head_prior:
            _hp = torch.from_numpy(np.load(head_prior_path)).float()
            assert _hp.shape == (num_ents, prior_n), (
                f"head_prior 形状 {_hp.shape} != (num_ents, prior_n) = ({num_ents}, {prior_n})")
            self.prior_lo, self.prior_n = int(prior_lo), int(prior_n)
            _zr = _hp[prior_lo:prior_lo + prior_n]
            assert float(_zr.abs().max()) < 1e-6, "卡型实体行应为全 0(不注入)"
            self.register_buffer("head_prior", _hp)
            self.head_alpha = torch.nn.Parameter(torch.tensor(float(head_prior_alpha_init)))
            _cov = float((_hp.max(dim=1).values > 0).float().mean())
            print(f"[head-prior] {head_prior_path} shape={tuple(_hp.shape)} "
                  f"cols=[{prior_lo},{prior_lo + prior_n})  alpha_init={head_prior_alpha_init}  "
                  f"有历史行的实体={_cov:.4f}  log1p 上界={float(_hp.max()):.3f}")

        # ── (H) 时序口径的先验表: 每快照 t 的表只统计【严格早于 t】的 train+valid 边 ──
        # 起因(§25.6 的探针): §23–§25 用的静态表是 train+valid 全历史, 于是训练/验证期每一行
        # 的 head 通道都非空 —— 它自己那条边就在表里(16.4% 的训练行在该 head 的任何更早快照上
        # 根本没有边, 而这些行 79.8% 的 argmax 正是它自己这行的 gold)。alpha_m 因此学在"这条
        # 通道常开且可信"的假象上(可用率 train 1.0000 vs test 0.1525), 而 val 也吃同一张假表。
        # 本块只改【训练/验证期看到哪张表】: 表按数据文件第 4 列的 ts 直接索引, 且 exporter 里
        # 已断言"所有 t>=首个 test 快照 的层与静态表逐位相同" => test 侧条件一个字都没变。
        # 635.9 MB 的表【不】走 register_buffer: 那是数据不是参数, 进 state_dict 会让每个
        # checkpoint 胖 0.6 GB(而分析脚本本来就按 npy 读它), 所以放普通属性并手动搬到 device。
        self.use_prior_ts = use_prior_ts
        if self.use_prior_ts:
            assert self.use_base_prior or self.use_head_prior, \
                "--prior-ts 需要至少一条输出层先验通道(--base-prior / --head-prior)"
            _T = None
            if self.use_base_prior:
                _bpt = torch.from_numpy(np.load(base_prior_ts_path)).float()
                assert _bpt.shape[1:] == (_pad + 1, prior_n), (
                    f"base_prior_ts 形状 {_bpt.shape} 的后两维 != (n_bases+1, prior_n) "
                    f"= ({_pad + 1}, {prior_n})")
                assert float(_bpt[:, _pad].abs().max()) < 1e-6, "每个 ts 的 pad 行都应是全 0"
                self.base_prior_ts = _bpt.to(self.device)
                _T = _bpt.shape[0]
                print(f"[prior-ts] {base_prior_ts_path} shape={tuple(_bpt.shape)} "
                      f"({self.base_prior_ts.numel() * 4 / 1e6:.0f} MB on {self.device})")
            if self.use_head_prior:
                _hpt = torch.from_numpy(np.load(head_prior_ts_path)).float()
                assert _hpt.shape[1:] == (num_ents, prior_n), (
                    f"head_prior_ts 形状 {_hpt.shape} 的后两维 != (num_ents, prior_n) "
                    f"= ({num_ents}, {prior_n})")
                assert float(_hpt[:, prior_lo:prior_lo + prior_n].abs().max()) < 1e-6, \
                    "每个 ts 的卡型实体行都应是全 0"
                self.head_prior_ts = _hpt.to(self.device)
                _T = _hpt.shape[0] if _T is None else _T
                assert _hpt.shape[0] == _T, "两张时序表的时间轴长度不一致"
                print(f"[prior-ts] {head_prior_ts_path} shape={tuple(_hpt.shape)} "
                      f"({self.head_prior_ts.numel() * 4 / 1e6:.0f} MB on {self.device})")
            self.prior_ts_len = int(_T)
            # 门控特征的时间轴归一常数(与静态口径同理: 除全表最大支持度)
            self._ts_norm_b = (float(self.base_prior_ts.sum(2).max())
                               if self.use_base_prior else 1.0)
            self._ts_norm_m = (float(self.head_prior_ts.sum(2).max())
                               if self.use_head_prior else 1.0)
            self._pending_ts = None
            # 首/末层与静态表的差 = 这次改动到底咬在哪: 训练期必须不同, test 段必须相同
            if self.use_base_prior and self.use_head_prior:
                # 静态表此刻还是 register_buffer 在 CPU 上(model.cuda() 在后面才调), 时序表已按
                # self.device 搬好 => 比较必须落到同一设备。
                assert not torch.equal(self.head_prior_ts[1].cpu(), self.head_prior), \
                    "ts=1 的时序表与静态表相同 => 时序口径没有生效"
                assert torch.equal(self.head_prior_ts[-1].cpu(), self.head_prior), \
                    "末层时序表应等于静态表(全历史), 否则 exporter 的累计口径错了"
                print(f"[prior-ts] 自检: ts=1 有 head 历史的实体="
                      f"{float((self.head_prior_ts[1].max(1).values > 0).float().mean()):.4f}  "
                      f"ts={self.prior_ts_len - 1} 有 head 历史的实体="
                      f"{float((self.head_prior_ts[-1].max(1).values > 0).float().mean()):.4f}  "
                      f"(静态表={float((self.head_prior.max(1).values > 0).float().mean()):.4f})")

        # ── §25 (F): 信任度从"两个全局标量"改成"按行的小门" ──
        # §24.6b 的定位: 全新 head 层那 −0.0306 不是 encoder 变坏(Q5 里 0.0439 > baseprior 0.0392),
        # 也不是标量取错了值, 而是"一个全局 alpha 没法按行分配" —— head 没历史的那批行(测试边 84.8%)
        # 必须把信任全给 base 通道, 但它跟"head 有历史"的行共用同一个 alpha_b, 于是训练时被后者拖着
        # 往下压(1.5557 -> 1.4006)。这里把 alpha 变成行的函数, 特征只用【已加载的两张表】自己导出:
        #   f1 = base 通道支持度(该 base 行 log1p 计数之和, 除全局最大) 
        #   f2 = head 通道支持度(该 head 行同上)          f3 = 1[f2 > 0]
        # 初值 w=0、c=softplus^-1(1)=0.54133 => 起点 alpha_b=alpha_m≡1.0, 与 §23/§24 的初值逐字相同,
        # 所以这一臂相对 §24 唯一的差别就是"能不能按行长", 不是起点不同。
        if self.use_prior_gate:
            assert self.use_base_prior or self.use_head_prior, \
                "--prior-gate 需要至少一条输出层先验通道(--base-prior / --head-prior)"
            self.gate_w = torch.nn.Parameter(torch.zeros(2, 3))
            self.gate_c = torch.nn.Parameter(torch.full((2,), float(np.log(np.expm1(1.0)))))
            if self.use_prior_ts:
                # 时序口径下"这个 head 有没有历史 / 支持度多大"是【实体 × ts】的函数, 不能按实体
                # 预先存 => 特征在 apply_base_prior 里按当前 ts 现算, 归一用全表最大值(标量)。
                print(f"[prior-gate] 时序口径: 特征按 (实体, ts) 现算  "
                      f"norm_b={self._ts_norm_b:.3f}  norm_m={self._ts_norm_m:.3f}  "
                      f"w=0, c={float(self.gate_c[0]):.5f} => 初始 alpha=1.0000 (两条通道同)")
            else:
                _gf = torch.zeros(num_ents, 3)
                if self.use_base_prior:
                    assert hasattr(self, "base_of"), "--prior-gate + --base-prior 需要 base_of"
                    _sb = self.base_prior.sum(dim=1)
                    _gf[:, 0] = _sb[self.base_of] / max(float(_sb.max()), 1e-6)
                if self.use_head_prior:
                    _sh = self.head_prior.sum(dim=1)
                    _gf[:, 1] = _sh / max(float(_sh.max()), 1e-6)
                    _gf[:, 2] = (_gf[:, 1] > 0).float()
                self.register_buffer("prior_gate_feat", _gf)
                print(f"[prior-gate] feat shape={tuple(_gf.shape)}  "
                      f"base 支持度>0 的实体={float(_gf[:, 0].gt(0).float().mean()):.4f}  "
                      f"head 支持度>0 的实体={float(_gf[:, 1].gt(0).float().mean()):.4f}  "
                      f"w=0, c={float(self.gate_c[0]):.5f} => 初始 alpha=1.0000 (两条通道同)")

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
                gpu_base=self.gpu_base,
                feat_dim_list=feat_dim_list,
                feat_alpha_fixed=self.decoder_feat_lambda,
            )
            self.rdecoder = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError

    # ============================================================
    def set_prior_ts(self, ts):
        """调用方在进入 predict/get_loss 之前说明"这批三元组属于哪个快照"。

        只有 --prior-ts(时序口径)需要; 静态口径下这是 no-op, 老调用点一个字不用改。
        """
        if getattr(self, "use_prior_ts", False):
            self._pending_ts = int(ts)

    def _pop_prior_ts(self):
        """取走本次调用的 ts 并清空 => 忘设 / 重复用同一个 ts 都会当场报错。"""
        if not getattr(self, "use_prior_ts", False):
            return None
        v, self._pending_ts = self._pending_ts, None
        assert v is not None, ("--prior-ts 已开启但调用方没有先 set_prior_ts(τ): "
                               "静默沿用上一个快照的表会让训练与评测读到不同时间面的先验")
        assert 0 <= v < self.prior_ts_len, f"ts={v} 越界时序表 [0, {self.prior_ts_len})"
        return v

    def _prior_tables(self, ts):
        """这批三元组该用的两张偏置表: 时序口径取第 ts 层, 静态口径取整表。"""
        if ts is None:
            return (self.base_prior if self.use_base_prior else None,
                    self.head_prior if self.use_head_prior else None)
        return (self.base_prior_ts[ts] if self.use_base_prior else None,
                self.head_prior_ts[ts] if self.use_head_prior else None)

    # ============================================================
    def apply_base_prior(self, score, triples, ts=None):
        """把历史卡型先验加到【候选列的 logit】上; 两个 flag 都关时原样返回(逐位同原版)。

        score:  [B, num_ents]  decoder 输出的候选打分(未过滤或已过滤都可以, -1e10 + 有限值仍最小)
        triples:[B, 3] (head, rel, tail); 只按 subject 给偏置 => 逆向行 subject 是卡型实体,
                base_of = pad、head_prior 该行全 0 => 偏置 0 行 => 该方向完全不受影响。
        ts:     --prior-ts 时的快照下标(= 数据文件第 4 列), 决定读哪一层表; 静态口径传 None。
        两项各自一个信任度, 关掉任一项时另一项行为与单通道版(§23)逐字相同。
        --prior-gate(§25) 时信任度取自按行的小门 softplus(w·f + c); 不开时仍是两个全局标量。
        """
        if not (self.use_base_prior or self.use_head_prior):
            return score
        sl = slice(self.prior_lo, self.prior_lo + self.prior_n)
        hd = triples[:, 0]
        bp, hp = self._prior_tables(ts)
        if self.use_prior_gate:
            if ts is None:                                   # 静态口径: 特征按实体预先存好
                _feat = self.prior_gate_feat[hd]
            else:                                            # 时序口径: 特征按 (实体, ts) 现算
                _z = torch.zeros(hd.numel(), device=score.device, dtype=score.dtype)
                if bp is not None:
                    _f0 = bp.sum(dim=1)[self.base_of[hd]] / max(self._ts_norm_b, 1e-6)
                else:
                    _f0 = _z
                if hp is not None:
                    _v = hp.sum(dim=1)[hd] / max(self._ts_norm_m, 1e-6)
                    _f1, _f2 = _v, (_v > 0).to(score.dtype)
                else:
                    _f1 = _f2 = _z
                _feat = torch.stack([_f0, _f1, _f2], dim=1)
            # [B, 2]: 第 0 列 = base 通道信任度, 第 1 列 = head 通道信任度(softplus => 非负)
            _al = torch.nn.functional.softplus(_feat @ self.gate_w.t() + self.gate_c)
            a_b, a_m = _al[:, :1], _al[:, 1:]
        else:
            a_b = self.prior_alpha if self.use_base_prior else None
            a_m = self.head_alpha if self.use_head_prior else None
        if self.use_base_prior:
            score[:, sl] = score[:, sl] + a_b * bp[self.base_of[hd]]
        if self.use_head_prior:
            score[:, sl] = score[:, sl] + a_m * hp[hd]
        return score

    # ============================================================
    def compute_dynamic_emb(self):
        if self.use_base_feat:
            # mask 让 pad 行(tail 段/无父 head)的梯度恒为 0, tail 表征逐位等于原版
            _inj = self.base_alpha * self.base_proj(self.base_feat[self.base_of])
            if self.use_base_emb:
                _inj = _inj + self.base_emb[self.base_of]
            return self.dynamic_emb + self.base_valid * _inj
        if self.use_base_emb:
            # mask 让 pad 行(tail 段/无父 head)的梯度恒为 0, tail 表征逐位等于原版
            return self.dynamic_emb + self.base_valid * self.base_emb[self.base_of]
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
            static_graph = static_graph.to(self.device)
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
            g = g.to(self.device)
            temp_e = self.h[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_rels * 2, self.h_dim).float()
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1], :]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean
            if self.use_rel_evolution:
                if i == 0:
                    x_input = torch.cat((self.emb_rel, x_input), dim=1)
                    self.h_0 = self.relation_cell_1(x_input, self.emb_rel)
                    self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
                else:
                    x_input = torch.cat((self.emb_rel, x_input), dim=1)
                    self.h_0 = self.relation_cell_1(x_input, self.h_0)
                    self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            else:
                # 消融①: 关系嵌入不随快照演化, 每步都用静态 emb_rel
                self.h_0 = self.emb_rel
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
            all_triples = all_triples.to(self.device)

            evolve_embs, _, r_emb, _, _ = self.forward(test_graph, static_graph, use_cuda)
            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
            # 时序口径: 这一整批 test/valid 三元组同属一个快照, 取一次 ts 给整个循环用
            _pts = self._pop_prior_ts()

            N = all_triples.size(0)
            rank_raw_list, rank_filter_list = [], []
            rank_raw_list_r, rank_filter_list_r = [], []

            # ── 候选集口径 ──
            # ent_segment = (lo, hi) 闭区间时，实体排名不再在全部 num_nodes 个实体里做，
            # 而是按【方向各自的类型段】做：正向 tail 段 [lo,hi]，逆向 head 段 [0,lo)。
            # 现行论文口径(主表)是 None = 全实体，两者必须在同一张表里都报出来，
            # 否则"模型不及闭式基线"这种结论会被候选空间宽度污染。
            _seg = getattr(self, "ent_segment", None)

            def _ent_rank(sc, trip):
                """sc: [B, num_nodes] 已过滤的分数；返回 0-based 实体排名。"""
                if _seg is None:
                    return sort_and_rank(sc, trip[:, 2])
                lo, hi = _seg
                out = torch.empty(sc.size(0), dtype=torch.long, device=sc.device)
                fwd = (trip[:, 1] < num_rels)
                # 用基本切片(不是 arange 收集)，逆向段少拷一次 [B, num_nodes]
                for mask, sub_slice, shift in ((fwd, slice(lo, hi + 1), lo),
                                               (~fwd, slice(0, lo), 0)):
                    if not mask.any():
                        continue
                    sub = sc[mask][:, sub_slice]
                    tgt = trip[mask, 2] - shift
                    if tgt.numel():
                        assert int(tgt.min()) >= 0 and int(tgt.max()) < sub.size(1), (
                            f"--ent-segment {lo} {hi} 容不下真值: "
                            f"gold∈[{int(trip[mask, 2].min())}, {int(trip[mask, 2].max())}] "
                            f"({'正向 tail 段' if shift else '逆向 head 段'}), "
                            f"候选段 [{lo}, {hi}]。请核对数据产物里的 segment。")
                    out[mask] = sort_and_rank(sub, tgt)
                return out

            for start in range(0, N, eval_bs):
                end = min(start + eval_bs, N)
                batch_triples = all_triples[start:end]

                score = self.decoder_ob.forward(embedding, r_emb, batch_triples, mode="test")
                score = self.apply_base_prior(score, batch_triples, _pts)   # §23: 输出层先验(默认关); §26 时序口径取 ts

                rank_raw_list.append(_ent_rank(score, batch_triples))

                # ── B7/hybrid 图通道导出: 【过滤前】GPU 段分数(纯只读, 默认不开) ──
                # 必须卡在 filter_score 之前: 过滤会把该 head 的其它已知答案列置 -inf,
                # 拿过滤后的分数当混合器特征 = 把真值侧信息泄进通道(§46 的恒等式教训)。
                # 探针里没有 "raw_seg" 这个 key 时本段整块跳过 ⇒ 论文路径逐位不动。
                _p0 = getattr(self, "seg_probe", None)
                if _p0 is not None and "raw_seg" in _p0:
                    _m0 = (batch_triples[:, 1] == 1)
                    if _m0.any():
                        _gb0, _nc0 = _p0["gpu_base"], _p0["n_cls"]
                        _bt0 = batch_triples[_m0]
                        _p0["raw_seg"].append(score[_m0][:, _gb0:_gb0 + _nc0].detach().float().cpu())
                        _p0["raw_head"].append(_bt0[:, 0].cpu())
                        _p0["raw_gold"].append(_bt0[:, 2].cpu())

                if all_ans_ent is not None:
                    filter_score(batch_triples, score, all_ans_ent)
                    rank_filter_list.append(_ent_rank(score, batch_triples))

                    # ── 可选探针(仅 r1 正向边): top-1 是否落在 GPU 段 + 段内排名 ──
                    # 用途: 把"124,891 路排序的 H@1"和"5 类分类 accuracy"放到同一口径下比较。
                    _p = getattr(self, "seg_probe", None)
                    if _p is not None:
                        _m = (batch_triples[:, 1] == 1)
                        if _m.any():
                            _gb, _nc = _p["gpu_base"], _p["n_cls"]
                            _bt = batch_triples[_m]
                            _sc = score[_m]
                            _cols = torch.arange(_gb, _gb + _nc, device=score.device)
                            _sub = _sc[:, _cols]                       # [n, 6] 段内分数(已过滤)
                            _ti = (_bt[:, 2] - _gb).clamp(0, _nc - 1)
                            _gs = _sub.gather(1, _ti.view(-1, 1))
                            _p["seg_rank"].append((_sub > _gs).sum(1).cpu())
                            _p["top1"].append(_sc.argmax(1).cpu())
                            _p["gold"].append(_bt[:, 2].cpu())

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
        loss_ent = _loss_accumulator(self.device)
        loss_rel = _loss_accumulator(self.device)
        loss_static = _loss_accumulator(self.device)

        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.device)

        evolve_embs, static_emb, r_emb, _, _ = self.forward(glist, static_graph, use_cuda)
        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
        # 时序口径: 这一整个 step 的三元组同属一个快照, 取一次 ts 给全部 batch 用
        _pts = self._pop_prior_ts()

        if self.entity_prediction:
            ent_eval_bs = 2000
            num_rows = all_triples.size(0)
            loss_ent_sum = torch.zeros((), device=all_triples.device, dtype=torch.float32)
            for start in range(0, num_rows, ent_eval_bs):
                end = min(start + ent_eval_bs, num_rows)
                batch_triples = all_triples[start:end]
                batch_scores = self.decoder_ob.forward(pre_emb, r_emb, batch_triples).view(-1, self.num_ents)
                batch_scores = self.apply_base_prior(batch_scores, batch_triples, _pts)  # §23: 训练与评测同一套偏置
                loss_ent_sum = loss_ent_sum + F.cross_entropy(batch_scores, batch_triples[:, 2], reduction="sum")
                del batch_scores
            loss_ent += loss_ent_sum / num_rows

        # ── compat 辅助分类 loss: 直接强监督 compat_head (5-GPU 分类) ──
        if self.use_compat and self.compat_aux_weight > 0:
            r1_fwd_mask = (all_triples[:, 1] == 1)  # 只正向 r1 边
            if r1_fwd_mask.any():
                r1_fwd = all_triples[r1_fwd_mask]
                h_ids = r1_fwd[:, 0]
                # tail 是 GPU 段实体 id, 换算成 compat_head 的 6 类索引(0=CPU..5=V100M32)
                gpu_labels = r1_fwd[:, 2] - self.gpu_base
                if gpu_labels.min() < 0 or gpu_labels.max() > 5:
                    raise RuntimeError(
                        f"compat aux label out of range: [{gpu_labels.min()}, {gpu_labels.max()}] "
                        f"(gpu_base={self.gpu_base}). 说明 id 空间判定与数据不一致, 请检查 main.py 的 "
                        f"[id-space] 输出与 --data-dir/--gpu-id-base 是否配套。")
                feat = self._compat_feat_algo[h_ids]
                compat_logits = self.compat_head(feat)  # [n_r1, 6]
                aux_loss = F.cross_entropy(compat_logits, gpu_labels,
                                           weight=self.compat_class_weights)
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

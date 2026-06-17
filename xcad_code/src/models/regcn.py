import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, self_loop=False, dropout=0.0):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases if num_bases > 0 else num_rels
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        if self.num_bases > 0 and self.num_bases < self.num_rels:
            self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat, self.out_feat))
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))
        else:
            self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.in_feat, self.out_feat))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            self.w_comp = None

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.bias)

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

    def msg_func(self, edges):
        if self.w_comp is not None and self.num_bases < self.num_rels:
            weight = torch.matmul(self.w_comp, self.weight.view(self.num_bases, -1)).view(
                self.num_rels, self.in_feat, self.out_feat)
        else:
            weight = self.weight

        w = weight.index_select(0, edges.data['type'])
        msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
        return {'msg': msg}

    def propagate(self, g):
        g.update_all(self.msg_func, fn.sum(msg='msg', out='h'))

    def forward(self, g, prev_h=None):
        if self.self_loop and 'h' in g.ndata:
            loop_msg = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout:
                loop_msg = self.dropout(loop_msg)

        self.propagate(g)
        node_repr = g.ndata['h']

        if self.self_loop:
            node_repr = node_repr + loop_msg

        if self.bias:
            node_repr = node_repr + self.bias

        if self.activation:
            node_repr = self.activation(node_repr)

        g.ndata['h'] = node_repr
        return node_repr


class RGCNCell(nn.Module):
    def __init__(self, num_nodes, h_dim, num_rels, num_bases=-1, dropout=0,
                 self_loop=False, encoder_name="rgcn"):
        super(RGCNCell, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.num_rels = num_rels
        self.encoder_name = encoder_name

        self.layer = RGCNLayer(
            h_dim, h_dim, num_rels, num_bases,
            activation=F.rrelu, self_loop=self_loop, dropout=dropout
        )

    def forward(self, g, init_ent_emb):
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = init_ent_emb[node_id]
        h = self.layer.forward(g)
        return h


class ConvTransE(nn.Module):
    def __init__(self, num_entities, embedding_dim, input_dropout=0, hidden_dropout=0,
                 feature_dropout=0, channels=50, kernel_size=3):
        super(ConvTransE, self).__init__()
        self.inp_drop = nn.Dropout(input_dropout)
        self.hidden_drop = nn.Dropout(hidden_dropout)
        self.feature_drop = nn.Dropout(feature_dropout)

        self.conv1 = nn.Conv1d(2, channels, kernel_size,
                               stride=1, padding=int(math.floor(kernel_size / 2)))
        self.bn0 = nn.BatchNorm1d(2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        self.fc = nn.Linear(embedding_dim * channels, embedding_dim)
        self.bn3 = nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(num_entities)))

    def forward(self, embedding, emb_rel, triplets, mode="train"):
        e1_embedded = F.tanh(embedding[triplets[:, 0]]).unsqueeze(1)
        rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)

        stacked = torch.cat([e1_embedded, rel_embedded], 1)
        stacked = self.bn0(stacked)
        x = self.inp_drop(stacked)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        score = torch.mm(x, embedding.t())
        return score


class REGCNModel(nn.Module):
    def __init__(self, num_nodes, num_rels, h_dim=64, n_layers=1, num_bases=-1,
                 dropout=0, self_loop=False, skip_connect=False, layer_norm=False,
                 input_dropout=0, hidden_dropout=0, feat_dropout=0):
        super(REGCNModel, self).__init__()

        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.h_dim = h_dim
        self.layer_norm = layer_norm

        self.emb_rel = nn.Parameter(torch.Tensor(num_rels * 2, h_dim), requires_grad=True)
        nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = nn.Parameter(torch.Tensor(num_nodes, h_dim), requires_grad=True)
        nn.init.normal_(self.dynamic_emb)

        self.rgcn_cell = RGCNCell(
            num_nodes, h_dim, num_rels * 2, num_bases,
            dropout=dropout, self_loop=self_loop
        )

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)

        self.relation_cell = nn.GRUCell(h_dim * 2, h_dim)

        self.decoder = ConvTransE(
            num_nodes, h_dim, input_dropout, hidden_dropout, feat_dropout
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, g_list, init_embs=None):
        if init_embs is None:
            h = F.normalize(self.dynamic_emb, p=2, dim=1) if self.layer_norm else self.dynamic_emb
        else:
            h = init_embs

        history_embs = []
        h_r = self.emb_rel

        for i, g in enumerate(g_list):
            g.ndata['id'] = g.ndata.get('id', torch.arange(self.num_nodes, device=h.device))

            edge_types = g.edata['type']
            unique_rels = torch.unique(edge_types)

            rel_context = torch.zeros(self.num_rels * 2, self.h_dim, device=h.device)
            for r_idx in unique_rels:
                mask = edge_types == r_idx
                if mask.any():
                    src_nodes = g.edges()[0][mask]
                    avg_emb = h[src_nodes].mean(dim=0)
                    rel_context[r_idx] = avg_emb

            x_input = torch.cat([h_r, rel_context], dim=1)
            if i == 0:
                h_r = self.relation_cell(x_input, self.emb_rel)
            else:
                h_r = self.relation_cell(x_input, h_r)

            h_r = F.normalize(h_r, p=2, dim=1) if self.layer_norm else h_r

            current_h = self.rgcn_cell.forward(g, h)

            current_h = F.normalize(current_h, p=2, dim=1) if self.layer_norm else current_h

            time_weight = torch.sigmoid(torch.mm(h, self.time_gate_weight) + self.time_gate_bias)
            h = time_weight * current_h + (1 - time_weight) * h

            history_embs.append(h)

        return history_embs, h_r

    def predict(self, embedding, r_emb, triplets):
        score = self.decoder.forward(embedding, r_emb, triplets)
        return score

    def get_loss(self, glist, triples, all_triples=None):
        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels

        if all_triples is None:
            all_triples = torch.cat([triples, inverse_triples])

        history_embs, r_emb = self.forward(glist)
        pre_emb = F.normalize(history_embs[-1], p=2, dim=1) if self.layer_norm else history_embs[-1]

        scores = self.decoder.forward(pre_emb, r_emb, all_triples)
        loss = self.loss_fn(scores, all_triples[:, 2])

        return loss, pre_emb, r_emb
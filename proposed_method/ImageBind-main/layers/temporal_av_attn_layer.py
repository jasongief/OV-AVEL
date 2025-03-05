#! this .py file heavily borrows from https://github.com/yujiangpu20/PEL4VAD/tree/master, thanks for their great codes
import torch
import torch.nn as nn
from torch import FloatTensor
from torch.nn.parameter import Parameter
from scipy.spatial.distance import pdist, squareform
import torch.nn.functional as F
import numpy as np
import math
from copy import deepcopy as dcp
import pdb

class DistanceAdj(nn.Module):
    def __init__(self, sigma=0.6, bias=0.2):
        super(DistanceAdj, self).__init__()
        # self.sigma = sigma
        # self.bias = bias
        #! not learnable
        # self.w = sigma
        # self.b = bias
        #! learnable
        self.w = nn.Parameter(torch.FloatTensor(1))
        self.b = nn.Parameter(torch.FloatTensor(1))
        self.w.data.fill_(sigma)
        self.b.data.fill_(bias)

    def forward(self, batch_size, seq_len, n_heads):
        arith = np.arange(seq_len).reshape(-1, 1)
        dist = pdist(arith, metric='cityblock').astype(np.float32)
        dist = torch.from_numpy(squareform(dist)).cuda()
        # dist = torch.from_numpy(squareform(dist))
        # dist = torch.exp(-self.sigma * dist ** 2) # no use
        dist = torch.exp(-torch.abs(self.w * dist ** 2 - self.b))
        dist = dist.unsqueeze(0).unsqueeze(0).repeat(batch_size, n_heads, 1, 1)

        return dist



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, dim_k, dim_v, n_heads, norm_flag=None):
        super(MultiHeadAttention, self).__init__()
        self.dim_v = dim_v
        self.dim_k = dim_k  # same as dim_q
        self.n_heads = n_heads
        self.norm_flag = norm_flag

        self.q = nn.Linear(d_model, dim_k)
        self.k = nn.Linear(d_model, dim_k)
        self.v = nn.Linear(d_model, dim_v)
        self.o = nn.Linear(dim_v, d_model)

        self.norm_fact = 1 / math.sqrt(dim_k)
        self.act = nn.Softmax(dim=-1)
        self.alpha = nn.Parameter(torch.FloatTensor([0]))


    def forward(self, query, key, value, mask=None, adj=None):
        # query, key, value: [bs, 10, 1024]
        bs = query.shape[0]
        Q = self.q(query).view(bs, -1, self.n_heads, self.dim_k // self.n_heads).transpose(1, 2) # [bs, 10, h, dim_k/h] -> [bs, h, 10, dim_k/h]
        K = self.k(key).view(bs, -1, self.n_heads, self.dim_k // self.n_heads).transpose(1, 2) # -> [bs, h, 10, dim_k/h]
        V = self.v(value).view(bs, -1, self.n_heads, self.dim_v // self.n_heads).transpose(1, 2) # -> [bs, h, 10, dim_v/h]

        if adj is not None:
            g_map = torch.matmul(Q, K.transpose(-2, -1)) * self.norm_fact + adj
        else:
            g_map = torch.matmul(Q, K.transpose(-2, -1)) * self.norm_fact # [bs, h, 10, 10]
        
        if mask is not None:
            l_map = g_map.clone()
            l_map = l_map.masked_fill_(mask.data.eq(0), -1e9)
        # pdb.set_trace()

        g_map = self.act(g_map) # [bs, h, 10, 10]
        glb = torch.matmul(g_map, V).transpose(1, 2).reshape(bs, query.shape[1], -1) # [bs, h, 10, dim_v/h] -> [bs, 10, h, dim_v/h] -> [bs, 10, dim_v]
        
        if mask is not None:
            l_map = self.act(l_map)
            lcl = torch.matmul(l_map, V).transpose(1, 2).reshape(bs, query.shape[1], -1) # -> [bs, 10, dim_v]
            alpha = torch.sigmoid(self.alpha)
            tmp = alpha * glb + (1 - alpha) * lcl
            # tmp = lcl
        else:
            tmp = glb
        # pdb.set_trace()

        if self.norm_flag:
            tmp = torch.sqrt(F.relu(tmp)) - torch.sqrt(F.relu(-tmp))  # power norm
            tmp = F.normalize(tmp)  # l2 norm
        # tmp = self.o(tmp).view(-1, x.shape[1], x.shape[2]) # [bs, 10, d_model]
        tmp = self.o(tmp) # [bs, 10, d_model]
        # pdb.set_trace()

        return tmp


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        # self.pre_norm = nn.LayerNorm(d_model)
        # self.post_norm = nn.LayerNorm(d_ff)

    def forward(self, x):
        # x = self.pre_norm(x)
        output = self.w_2(self.dropout(F.relu(self.w_1(x))))
        # output = self.post_norm(output)
        return output


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model=1024,
        hid_dim=256,
        d_ff = 512,
        # out_dim,
        n_heads=1,
        dropout=0.1,
        use_adj_in_attn=False,
        gamma=0.6,
        bias=0.2,
        use_mask_in_attn=False,
        win_size=1,
        norm_flag=None
    ):
        super(TransformerEncoder, self).__init__()
        self.n_heads = n_heads
        self.win_size = win_size
        self.self_attn = MultiHeadAttention(d_model, hid_dim, hid_dim, n_heads, norm_flag)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        # self.linear1 = nn.Conv1d(d_model, d_model // 2, kernel_size=1)
        # self.linear2 = nn.Conv1d(d_model // 2, out_dim, kernel_size=1)
        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.loc_adj = DistanceAdj(gamma, bias)
        self.use_adj_in_attn = use_adj_in_attn
        self.use_mask_in_attn = use_mask_in_attn

    def forward(self, q, k, v):
        # q, k, v: [bs, 10, 1024] 
        adj = None
        if self.use_adj_in_attn:
            adj = self.loc_adj(q.shape[0], q.shape[1], self.n_heads) # [bs, n_heads, 10, 10]
        mask = None
        if self.use_mask_in_attn:
            mask = self.get_mask(self.win_size, q.shape[1], q.shape[0])
        # Multi-Head Attention, post-LayerNorm is used.
        x = q + self.self_attn(q, k, v, mask, adj) # [bs, 10, d_model]
        x = self.norm1(x) # [bs, 10, d_model]
        # pdb.set_trace()

        # Feed-Forward Network, post-LayerNorm is used.
        x = x + self.ffn(x)
        x = self.norm2(x) # [bs, 10, d_model]
        # pdb.set_trace()

        # x = self.dropout1(F.gelu(self.linear1(x)))
        # x_e = self.dropout2(F.gelu(self.linear2(x)))
        return x
        # return x_e, x


    def get_mask(self, window_size, temporal_scale, bs):
        m = torch.zeros((temporal_scale, temporal_scale))
        w_len = window_size
        for j in range(temporal_scale):
            for k in range(w_len):
                m[j, min(max(j - w_len // 2 + k, 0), temporal_scale - 1)] = 1.

        m = m.repeat(bs, self.n_heads, 1, 1).cuda() # [bs, n_head, 10, 10]
        # m = m.repeat(bs, self.n_heads, 1, 1) # [bs, n_head, 10, 10]
        return m


class TemporalAttentionModule(nn.Module):
    def __init__(
        self,
        sa_layer_num=1,
        xa_layer_num=1,
        feat_dim=1024,
        hid_dim=256,
        d_ff=512,
        head_num=1,
        dropout=0.1,
        use_adj_in_attn=False,
        gamma=0.6,
        bias=0.2,
        use_mask_in_attn=False,
        win_size=1,
        norm_flag=None
    ):
        super(TemporalAttentionModule, self).__init__()
        block = TransformerEncoder(
            d_model=feat_dim,
            hid_dim=hid_dim,
            d_ff=d_ff,
            n_heads=head_num,
            dropout=dropout,
            use_adj_in_attn=use_adj_in_attn,
            gamma=gamma,
            bias=bias,
            use_mask_in_attn=use_mask_in_attn,
            win_size=win_size,
            norm_flag=norm_flag
        )

        #! parameter-shared in each layer and modality
        # self.a_sa_blocks = nn.Sequential(*[block for i in range(sa_layer_num)])
        # self.a_xa_blocks = nn.Sequential(*[block for i in range(xa_layer_num)])
        # self.v_sa_blocks = nn.Sequential(*[block for i in range(sa_layer_num)])
        # self.v_xa_blocks = nn.Sequential(*[block for i in range(xa_layer_num)])
        
        #! update 2024.10.26, correct version for parameter unshared
        self.a_sa_blocks = nn.Sequential(*[dcp(block) for i in range(sa_layer_num)])
        self.a_xa_blocks = nn.Sequential(*[dcp(block) for i in range(xa_layer_num)])
        self.v_sa_blocks = nn.Sequential(*[dcp(block) for i in range(sa_layer_num)])
        self.v_xa_blocks = nn.Sequential(*[dcp(block) for i in range(xa_layer_num)])
        # self.alpha_a = nn.Parameter(torch.FloatTensor([0.]))
        # self.alpha_v = nn.Parameter(torch.FloatTensor([0.]))
        

    def forward(self, a_tokens, v_tokens):
        # pdb.set_trace()

        # ori_a, ori_v = a_tokens, v_tokens
        for blk_id, blk in enumerate(self.a_sa_blocks):
            a_tokens = blk(a_tokens, a_tokens, a_tokens)
        a_sa_tokens = a_tokens
        # pdb.set_trace()
        
        # a_tokens = ori_a
        # for blk_id, blk in enumerate(self.a_xa_blocks):
        #     a_tokens = blk(a_tokens, v_tokens, v_tokens)
        # a_xa_tokens = a_tokens
        # pdb.set_trace()


        for blk_id, blk in enumerate(self.v_sa_blocks):
            v_tokens = blk(v_tokens, v_tokens, v_tokens)
        v_sa_tokens = v_tokens
        # pdb.set_trace()

        # v_tokens = ori_v
        # a_tokens = ori_a
        # for blk_id, blk in enumerate(self.v_xa_blocks):
        #     v_tokens = blk(v_tokens, a_tokens, a_tokens)
        # v_xa_tokens = v_tokens
        # pdb.set_trace()
        
        #! ablation: not learnable
        # a_tokens = a_sa_tokens + a_xa_tokens
        # v_tokens = v_sa_tokens + v_xa_tokens
        # a_tokens = (a_sa_tokens + a_xa_tokens) * 0.5
        # v_tokens = (v_sa_tokens + v_xa_tokens) * 0.5

        #! only self attention
        a_tokens = a_sa_tokens
        v_tokens = v_sa_tokens
        #! only cross attention
        # a_tokens = a_xa_tokens
        # v_tokens = v_xa_tokens


        #! learnable, turn to nan
        # alpha_a = torch.sigmoid(self.alpha_a)
        # a_tokens = alpha_a * a_sa_tokens + (1 - alpha_a) * a_xa_tokens
        # alpha_v = torch.sigmoid(self.alpha_v)
        # v_tokens = alpha_v * v_sa_tokens + (1 - alpha_v) * v_xa_tokens
        # # pdb.set_trace()

        if  torch.sum(a_tokens.isnan()) > 0 or torch.sum(v_tokens.isnan()) > 0:
            pdb.set_trace()

        return a_tokens, v_tokens




class TemporalLinearModule(nn.Module):
    def __init__(
        self,
        sa_layer_num=1,
        xa_layer_num=1,
        feat_dim=1024,
        hid_dim=256,
        d_ff=512,
        head_num=1,
        dropout=0.1,
        use_adj_in_attn=False,
        gamma=0.6,
        bias=0.2,
        use_mask_in_attn=False,
        win_size=1,
        norm_flag=None
    ):
        super(TemporalLinearModule, self).__init__()
        block = nn.Linear(feat_dim, feat_dim)

        # self.a_sa_blocks = nn.Sequential(*[block for i in range(sa_layer_num)])
        # self.a_xa_blocks = nn.Sequential(*[block for i in range(xa_layer_num)])
        # self.v_sa_blocks = nn.Sequential(*[block for i in range(sa_layer_num)])
        # self.v_xa_blocks = nn.Sequential(*[block for i in range(xa_layer_num)])
        
        #! update 2024.10.26, correct version for parameter unshared
        self.a_sa_blocks = nn.Sequential(*[dcp(block) for i in range(sa_layer_num)])
        self.a_xa_blocks = nn.Sequential(*[dcp(block) for i in range(xa_layer_num)])
        self.v_sa_blocks = nn.Sequential(*[dcp(block) for i in range(sa_layer_num)])
        self.v_xa_blocks = nn.Sequential(*[dcp(block) for i in range(xa_layer_num)])
        

        # self.alpha_a = nn.Parameter(torch.FloatTensor([0.]))
        # self.alpha_v = nn.Parameter(torch.FloatTensor([0.]))
        

    def forward(self, a_tokens, v_tokens):
        # pdb.set_trace()

        # ori_a, ori_v = a_tokens, v_tokens
        for blk_id, blk in enumerate(self.a_sa_blocks):
            a_tokens = blk(a_tokens)
        a_sa_tokens = a_tokens
        # pdb.set_trace()
        
        # a_tokens = ori_a
        # for blk_id, blk in enumerate(self.a_xa_blocks):
        #     a_tokens = blk(a_tokens, v_tokens, v_tokens)
        # a_xa_tokens = a_tokens
        # pdb.set_trace()


        for blk_id, blk in enumerate(self.v_sa_blocks):
            v_tokens = blk(v_tokens)
        v_sa_tokens = v_tokens
        # pdb.set_trace()

        # v_tokens = ori_v
        # a_tokens = ori_a
        # for blk_id, blk in enumerate(self.v_xa_blocks):
        #     v_tokens = blk(v_tokens, a_tokens, a_tokens)
        # v_xa_tokens = v_tokens
        # pdb.set_trace()
        
        #! ablation: not learnable
        # a_tokens = a_sa_tokens + a_xa_tokens
        # v_tokens = v_sa_tokens + v_xa_tokens
        # a_tokens = (a_sa_tokens + a_xa_tokens) * 0.5
        # v_tokens = (v_sa_tokens + v_xa_tokens) * 0.5
        #! only self attention
        a_tokens = a_sa_tokens
        v_tokens = v_sa_tokens
        #! only cross attention
        # a_tokens = a_xa_tokens
        # v_tokens = v_xa_tokens


        #! learnable, turn to nan
        # alpha_a = torch.sigmoid(self.alpha_a)
        # a_tokens = alpha_a * a_sa_tokens + (1 - alpha_a) * a_xa_tokens
        # alpha_v = torch.sigmoid(self.alpha_v)
        # v_tokens = alpha_v * v_sa_tokens + (1 - alpha_v) * v_xa_tokens
        # # pdb.set_trace()

        if  torch.sum(a_tokens.isnan()) > 0 or torch.sum(v_tokens.isnan()) > 0:
            pdb.set_trace()

        return a_tokens, v_tokens



if __name__ == "__main__":
    temporal_scale = 10
    window_size = 4
    bs = 10
    n_heads = 8
    m = torch.zeros((temporal_scale, temporal_scale))
    w_len = window_size
    for j in range(temporal_scale):
        for k in range(w_len):
            m[j, min(max(j - w_len // 2 + k, 0), temporal_scale - 1)] = 1.
    # pdb.set_trace()
    m = m.repeat(bs, n_heads, 1, 1)

    a_tokens = torch.randn(2, 10, 1024)
    v_tokens = torch.randn(2, 10, 1024)
    tam = TemporalAttentionModule(
        sa_layer_num=2,
        xa_layer_num=2,
        feat_dim=1024,
        hid_dim=256,
        d_ff=512,
        head_num=n_heads,
        dropout=0.1,
        use_adj_in_attn=True,
        gamma=0.6,
        bias=0.2,
        use_mask_in_attn=True,
        win_size=4,
        norm_flag=None
    )
    new_a, new_v = tam(a_tokens, v_tokens)


    pdb.set_trace()
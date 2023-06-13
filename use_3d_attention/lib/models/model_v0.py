import torch
import torch.nn.functional as F
from torch import nn, einsum
import numpy as np
from einops import rearrange

"""
没有encoder的过程
"""


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def ff_encodings(x, B):
    x_proj = (2. * np.pi * x.unsqueeze(-1)) @ B.t()
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4.0, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, int(dim * mult * 2)),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mult), dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, a2, __ = x.shape
        scale = a2 ** -0.5
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * scale
        # sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


class RowColTransformer(nn.Module):
    def __init__(self, num_tokens, dim, nfeats, depth, dim_head, attn_dropout, ff_dropout, style='col',
                 device=None, each_day_feature_num=0, each_day_cat_feature_num=0, sequence_length=5):
        super().__init__()
        self.device = device
        self.embeds = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])
        self.layers_mirror = nn.ModuleList([])
        self.mask_embed = nn.Embedding(nfeats, dim)
        max_length = 10
        self.each_day_feature_num = each_day_feature_num
        self.each_day_cat_feature_num = each_day_cat_feature_num
        self.pos_embedding = nn.Embedding(max_length, int(dim * nfeats / 5))
        self.scale = torch.sqrt(torch.FloatTensor([dim * nfeats / 5]).to(device))
        self.style = style
        self.sequence_length = sequence_length
        # dim= 6
        encoder_num = dim
        self.dim = dim
        hidden_size=512
        num_layers=5
        self.simple_MLP = nn.ModuleList(
            [simple_MLP([hidden_size, self.each_day_feature_num*2, self.each_day_feature_num]) for _ in range(sequence_length)])
        self.lstm = nn.LSTM(each_day_feature_num, hidden_size, num_layers=num_layers, batch_first=True)
        for _ in range(depth):
            if self.style == 'colrow':

                self.layers.append(nn.ModuleList([

                    PreNorm(each_day_feature_num, Residual(Attention(each_day_feature_num, heads=1, dropout=attn_dropout))),
                    PreNorm(each_day_feature_num, Residual(FeedForward(each_day_feature_num, dropout=ff_dropout))),
                    PreNorm(each_day_feature_num, Residual(Attention(each_day_feature_num, heads=1, dropout=attn_dropout))),
                    PreNorm(each_day_feature_num, Residual(FeedForward(each_day_feature_num, dropout=ff_dropout))),
                    PreNorm(sequence_length, Residual(Attention(sequence_length, heads=8, dropout=attn_dropout))),
                    PreNorm(sequence_length, Residual(FeedForward(sequence_length, dropout=ff_dropout))),
                    PreNorm(hidden_size, Residual(Attention(hidden_size, heads=8, dropout=attn_dropout))),
                    PreNorm(hidden_size, Residual(FeedForward(hidden_size, dropout=ff_dropout))),
                    nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True),
                    # nn.LSTM(hidden_size, each_day_feature_num, num_layers=num_layers, batch_first=True),
                    # PreNorm(encoder_num, Residual(Attention(encoder_num, heads=1, dropout=attn_dropout))),
                    # PreNorm(encoder_num, Residual(FeedForward(encoder_num, mult=1, dropout=ff_dropout)))
                ]
                ))
            else:

                self.layers.append(nn.ModuleList([
                    PreNorm(dim * nfeats,
                            Residual(Attention(dim * nfeats, heads=1, dim_head=64, dropout=attn_dropout))),
                    PreNorm(dim * nfeats, Residual(FeedForward(dim * nfeats, dropout=ff_dropout))),
                ]))

    def forward(self, x, x_cat=None):
        x = rearrange(x, 'b (d f) -> b d f',d=5)
        x_bat = x.clone()
        batch, n,f = x.shape
        # x_cont_dec = torch.ones((n, f, 512 - batch)).to(self.device)
        if self.style == 'colrow':
            # x, _ = self.lstm(x)
            for attn1_ff1s in self.layers:
                # x : 21 5 38
                # x1_1 = attn1_ff1s[0](x)
                # x1_2 = attn1_ff1s[1](x1_1)
                x1_3 = x.permute(1, 0, 2)
                x2_1 = attn1_ff1s[2](x1_3)
                x2_2 = attn1_ff1s[3](x2_1)
                x2_3 = x2_2.permute(1, 0, 2)
                x3_1 = x2_3.permute(0, 2, 1)
                x3_2 = attn1_ff1s[4](x3_1)
                x3_3 = attn1_ff1s[5](x3_2)
                x4_1 = x3_3.permute(0, 2, 1)
                x4_2 = attn1_ff1s[0](x4_1)
                x = attn1_ff1s[1](x4_2)
                # x4_1 = x2_3.permute(1, 2, 0)


                # x_cont_dec[:,:,0:batch] = x4_1
                # x4_1=torch.cat((x_cont_dec,x4_1),dim=2)
                # x4_2 = attn1_ff1s[6](x4_1)
                # x4_3 = attn1_ff1s[7](x4_2)
                # x4_3= x4_3[:,:,0:batch]
                # x = x4_3.permute( 2, 0,1)

                # x_, _ = attn1_ff1s[8](x3_4)
                # x, _ = attn1_ff1s[8](x2_3,_)
                # print(hn.shape)
                # x5_1 = rearrange(lstm_out, 'b d f -> (b d) f')
                # n1, n2 ,n3= lstm_out.shape
                # x_cont_dec_2 = torch.zeros(n1, n2 ,f).to(self.device)
                # for i in range(n2):
                #     x_cont_dec_2[:, i,:] = self.simple_MLP[i](lstm_out[:, i, :])
                # # x = rearrange(x_cont_dec, '(b d) f -> b d f', d=5)
                # # predictions = self.linear(lstm_out[:, -1, :])
                # x = x_cont_dec_2



        else:
            pass

        x, _ = self.lstm(x)
        x = (torch.cat((x,x_bat),2))[:, -1:, :]
        # x = x[:, -1:, :]
        # x = x[:, 0:1, :]
        # x = rearrange(x, 'b n d -> b (n d)')
        return x


# transformer
class Transformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
            ]))

    def forward(self, x, x_cont=None):
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


# mlp
class MLP(nn.Module):
    def __init__(self, dims, act=None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue
            if act is not None:
                layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class simple_MLP(nn.Module):
    def __init__(self, dims):
        super(simple_MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2])
        )

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.reshape(x.size(0), -1)
        x = self.layers(x)
        return x

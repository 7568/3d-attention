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

        for _ in range(depth):
            if self.style == 'colrow':
                self.layers.append(nn.ModuleList([
                    nn.ModuleList(
                        [PreNorm(encoder_num, Residual(Attention(encoder_num, heads=1, dropout=attn_dropout))),
                         PreNorm(encoder_num, Residual(FeedForward(encoder_num, dropout=ff_dropout))),
                         PreNorm(encoder_num, Residual(Attention(encoder_num, heads=1, dropout=attn_dropout))),
                         PreNorm(encoder_num, Residual(FeedForward(encoder_num, dropout=ff_dropout))),
                         PreNorm(encoder_num, Residual(Attention(encoder_num, heads=1, dropout=attn_dropout))),
                         PreNorm(encoder_num, Residual(FeedForward(encoder_num, dropout=ff_dropout))),
                         PreNorm(encoder_num, Residual(Attention(encoder_num, heads=1, dropout=attn_dropout))),
                         PreNorm(encoder_num, Residual(FeedForward(encoder_num, dropout=ff_dropout))),
                         PreNorm(encoder_num, Residual(Attention(encoder_num, heads=1, dropout=attn_dropout))),
                         PreNorm(encoder_num, Residual(FeedForward(encoder_num, dropout=ff_dropout))),
                         PreNorm(encoder_num, Residual(Attention(encoder_num, heads=1, dropout=attn_dropout))),
                         PreNorm(encoder_num, Residual(FeedForward(encoder_num, dropout=ff_dropout))),
                         PreNorm(encoder_num, Residual(Attention(encoder_num, heads=1, dropout=attn_dropout))),
                         PreNorm(encoder_num, Residual(FeedForward(encoder_num, mult=1, dropout=ff_dropout)))]),

                    nn.ModuleList([PreNorm(encoder_num * each_day_feature_num, Residual(
                        Attention(encoder_num * each_day_feature_num, heads=each_day_feature_num,
                                  dropout=attn_dropout))),
                                   PreNorm(encoder_num * each_day_feature_num, Residual(
                                       FeedForward(encoder_num * each_day_feature_num, dropout=ff_dropout))),
                                   PreNorm(encoder_num * each_day_feature_num, Residual(
                                       Attention(encoder_num * each_day_feature_num, heads=each_day_feature_num,
                                                 dropout=attn_dropout))),
                                   PreNorm(encoder_num * each_day_feature_num, Residual(
                                       FeedForward(encoder_num * each_day_feature_num, dropout=ff_dropout))),
                                   ]),

                    nn.ModuleList([PreNorm(encoder_num * sequence_length, Residual(
                        Attention(encoder_num * sequence_length, heads=sequence_length, dropout=attn_dropout))),
                                   PreNorm(encoder_num * sequence_length,
                                           Residual(FeedForward(encoder_num * sequence_length, dropout=ff_dropout))),
                                   PreNorm(encoder_num * sequence_length, Residual(
                                       Attention(encoder_num * sequence_length, heads=sequence_length,
                                                 dropout=attn_dropout))),
                                   PreNorm(encoder_num * sequence_length,
                                           Residual(FeedForward(encoder_num * sequence_length, dropout=ff_dropout)))
                                   ])

                ]))

            else:
                self.layers.append(nn.ModuleList([
                    PreNorm(dim * nfeats,
                            Residual(Attention(dim * nfeats, heads=1, dim_head=64, dropout=attn_dropout))),
                    PreNorm(dim * nfeats, Residual(FeedForward(dim * nfeats, dropout=ff_dropout))),
                ]))

    def forward(self, x, x_cat=None):
        if x_cat is not None:
            batch, n, _ = x.shape
            batch_c, n_c, _ = x_cat.shape
            x_new = []
            for i in range(self.sequence_length):
                x_new.append(torch.cat((x[:, i * (n // self.sequence_length):(i + 1) * (n // self.sequence_length), :],
                                        x_cat[:,
                                        i * (n_c // self.sequence_length):(i + 1) * (n_c // self.sequence_length), :]),
                                       dim=1))
            x = torch.cat(x_new, dim=1)

        batch, n, f = x.shape
        # x = rearrange(x, 'b (d f) -> b d f',d=5)
        if self.style == 'colrow':
            for attn1_ff1s, attn2_ff2s, attn3_ff3s in self.layers:
                # x : 21 (5 6) 38
                x1_1 = attn1_ff1s[0](x)
                x1 = attn1_ff1s[1](x1_1)
                x1_1 = rearrange(x, 'b (s d) f -> (s d) b f', s=5)  # 21 (5 6) 38 -> (5 6) 21 38
                x1_1 = attn1_ff1s[2](x1_1)
                x1_1 = attn1_ff1s[3](x1_1)
                x2 = rearrange(x1_1, '(s d) b f -> b (s d) f', s=5)
                x1_2 = rearrange(x, 'b (s d) f -> (b s) d f', s=5)  # (5 6) 21 38 -> (21 5) 6 38
                x1_2 = attn1_ff1s[4](x1_2)
                x1_2 = attn1_ff1s[5](x1_2)
                x3 = rearrange(x1_2, '(b s) d f -> b (s d) f', s=5)
                x1_3 = rearrange(x, 'b (s d) f -> d (b s) f', s=5)  # (21 5) 6 38 -> 6 (21 5) 38
                x1_3 = attn1_ff1s[6](x1_3)
                x1_3 = attn1_ff1s[7](x1_3)
                x4 = rearrange(x1_3, 'd (b s) f -> b (s d) f', s=5)
                x1_4 = rearrange(x, 'b (s d) f -> (b d) s f', s=5)  # 6 (21 5) 38 -> (21 6) 5 38
                x1_4 = attn1_ff1s[8](x1_4)
                x1_4 = attn1_ff1s[9](x1_4)
                x5 = rearrange(x1_4, '(b d) s f -> b (s d) f', b=batch)
                x1_5 = rearrange(x, 'b (s d) f -> s (b d) f', s=5)  # (21 6) 5 38 -> 5 (21 6) 38
                x1_5 = attn1_ff1s[10](x1_5)
                x1_5 = attn1_ff1s[11](x1_5)
                x6 = rearrange(x1_5, 's (b d) f -> b (s d) f', b=batch)
                x_ = torch.cat((x1,x2,x3,x4,x5,x6), dim=1)
                x_ = attn1_ff1s[12](x_)
                x_ = attn1_ff1s[13](x_)
                x = x_[:, 0:x6.shape[1], :]
                # x2_1 = rearrange(x1, 'b (s d) f -> s b (d f)', s=5)  # (21 5) 6 38 -> 5 21 (6 38)
                # x2_1 = attn2_ff2s[0](x2_1)
                # x2_1 = attn2_ff2s[1](x2_1)
                # x2 = rearrange(x2_1, 's b (d f) -> b (s d) f', f=f)
                # x2_2 = rearrange(x1, 'b (s d) f -> b s (d f)', s=5)  # 5 21 (6 38) -> 21 5 (6 38)
                # x2_2 = attn2_ff2s[2](x2_2)
                # x2_2 = attn2_ff2s[3](x2_2)
                # x2 += rearrange(x2_2, 'b s (d f) -> b (s d) f', f=f)
                # x2 /= 2.0
                #
                # x3_1 = rearrange(x2, 'b (s d) f -> b d (s f)', s=5)  # 5 21 (6 38) -> 21 6 (5 38)
                # x3_1 = attn3_ff3s[0](x3_1)
                # x3_1 = attn3_ff3s[1](x3_1)
                # x3 = rearrange(x3_1, 'b d (s f) -> b (s d) f', f=f)
                # x3_2 = rearrange(x2, 'b (s d) f-> d b (s f)', s=5)  # 21 6 (5 38) -> 6 21 (5 38)
                # x3_2 = attn3_ff3s[2](x3_2)
                # x3_2 = attn3_ff3s[3](x3_2)
                # x3 += rearrange(x3_2, 'd b (s f) -> b (s d) f', s=5)
                # x3 /= 2.0
                # x = x1

        else:
            for attn1, ff1 in self.layers:
                x = rearrange(x, 'b n d -> 1 b (n d)')
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, '1 b (n d) -> b n d', n=n)

        x = x[:, 0:1, :]
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

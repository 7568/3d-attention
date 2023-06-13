from .model_v0 import *


class sep_MLP(nn.Module):
    def __init__(self, dim, len_feats, categories):
        super(sep_MLP, self).__init__()
        self.len_feats = len_feats
        self.layers = nn.ModuleList([])
        for i in range(len_feats):
            self.layers.append(simple_MLP([dim, 5 * dim, categories[i]]))

    def forward(self, x):
        y_pred = list([])
        for i in range(self.len_feats):
            x_i = x[:, i, :]
            pred = self.layers[i](x_i)
            y_pred.append(pred)
        return y_pred


class ATTENTION_3D(nn.Module):
    def __init__(
            self,
            *,
            categories,
            num_continuous,
            dim,
            depth,
            dim_head=16,
            dim_out=1,
            mlp_hidden_mults=(4, 2),
            mlp_act=None,
            num_special_tokens=0,
            attn_dropout=0.,
            ff_dropout=0.,
            cont_embeddings='MLP',
            scalingfactor=10,
            attentiontype='col',
            final_mlp_style='common',
            y_dim=2,device=None,
            each_day_feature_num=0, each_day_cat_feature_num=0
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]

        self.register_buffer('categories_offset', categories_offset)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style

        if self.cont_embeddings == 'MLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1, 100, self.dim]) for _ in range(self.num_continuous)])
            input_size = (dim * self.num_categories) + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        elif self.cont_embeddings == 'pos_singleMLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1, 100, self.dim]) for _ in range(1)])
            input_size = (dim * self.num_categories) + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            print('Continous features are not passed through attention')
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories

            # transformer
        if attentiontype == 'col':
            self.transformer = Transformer(
                num_tokens=self.total_tokens,
                dim=dim,
                depth=depth,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout
            )
        elif attentiontype in ['row', 'colrow']:
            self.transformer = RowColTransformer(
                num_tokens=self.total_tokens,
                dim=dim,
                nfeats=nfeats,
                depth=depth,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                style=attentiontype,
                device=device,
                each_day_feature_num=each_day_feature_num,
                each_day_cat_feature_num=each_day_cat_feature_num
            )

        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act=mlp_act)
        self.embeds = nn.Embedding(self.total_tokens, self.dim)  # .to(device)

        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value=0)
        cat_mask_offset = cat_mask_offset.cumsum(dim=-1)[:-1]

        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value=0)
        con_mask_offset = con_mask_offset.cumsum(dim=-1)[:-1]

        self.register_buffer('cat_mask_offset', cat_mask_offset)
        self.register_buffer('con_mask_offset', con_mask_offset)

        self.mask_embeds_cat = nn.Embedding(self.num_categories * 2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous * 2, self.dim)
        self.single_mask = nn.Embedding(2, self.dim)
        self.pos_encodings = nn.Embedding(self.num_categories + self.num_continuous, self.dim)

        if self.final_mlp_style == 'common':
            self.mlp1 = simple_MLP([dim, (self.total_tokens) * 2, self.total_tokens])
            self.mlp2 = simple_MLP([dim, (self.num_continuous), 1])

        else:
            self.mlp1 = sep_MLP(dim, self.num_categories//5, categories[:self.num_categories//5])
            self.mlp2 = sep_MLP(dim, self.num_continuous//5, np.ones(self.num_continuous).astype(int))

        # self.mlpfory = simple_MLP([35, 128, y_dim])
        # self.mlpfory = simple_MLP([35+512, 128, y_dim])
        self.mlpfory = nn.Linear(35+512, y_dim)
        self.pt_mlp = simple_MLP([dim * (self.num_continuous + self.num_categories)//5,
                                  6 * dim * (self.num_continuous + self.num_categories) // 5,
                                  dim * (self.num_continuous + self.num_categories) // 2])
        self.pt_mlp2 = simple_MLP([dim * (self.num_continuous + self.num_categories)//5,
                                   6 * dim * (self.num_continuous + self.num_categories) // 5,
                                   dim * (self.num_continuous + self.num_categories) // 2])

    def forward(self, x_categ, x_cont):

        x = self.transformer(x_categ, x_cont)
        # x = rearrange(x, 'b (n d) -> b n d',d=self.dim)
        cat_outs = self.mlp1(x[:, :self.num_categories//5, :])
        con_outs = self.mlp2(x[:, self.num_categories//5:, :])
        return cat_outs, con_outs

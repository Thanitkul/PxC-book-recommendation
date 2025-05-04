import torch
from torch import nn
from typing import List, Dict, Tuple
class TwoTowerNet(nn.Module):
    def __init__(
        self,
        author_vocab_size: int,
        lang_vocab_size: int,
        tag_vocab_size: int,
        embed_dim: int,
        user_hidden_dims: List[int],
        item_hidden_dims: List[int],
        max_hist_len: int
    ):
        super().__init__()
        self.embed_dim     = embed_dim
        self.max_hist_len  = max_hist_len

        # embeddings for the 7 sparse features: [author, lang, tag1..tag5]
        self.author_emb = nn.Embedding(author_vocab_size, embed_dim, padding_idx=0)
        self.lang_emb   = nn.Embedding(lang_vocab_size, embed_dim, padding_idx=0)
        self.tag_emb    = nn.Embedding(tag_vocab_size, embed_dim, padding_idx=0)

        # user tower MLP: input = (7 * embed_dim * max_hist_len*2) + (3 * max_hist_len*2)
        user_input_dim = (7 * embed_dim) * (max_hist_len*2) + 3*(max_hist_len*2)
        user_layers = []
        prev = user_input_dim
        for h in user_hidden_dims:
            user_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        user_layers += [nn.Linear(prev, embed_dim)]
        self.user_mlp = nn.Sequential(*user_layers)

        # item tower MLP: input = 7*embed_dim + 3
        item_input_dim = 7*embed_dim + 3
        item_layers = []
        prev = item_input_dim
        for h in item_hidden_dims:
            item_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        item_layers += [nn.Linear(prev, embed_dim)]
        self.item_mlp = nn.Sequential(*item_layers)

    def embed_sparse(self, bids: torch.LongTensor):
        # bids: [batch_size] of book_id
        # gather author, lang, tag1..5
        a = self.author_emb(bids)
        l = self.lang_emb(bids)
        tags = self.tag_emb(bids)  # shape [batch,5,embed_dim]
        # flatten to [batch, 7*embed_dim]
        return torch.cat([a, l, tags.view(bids.size(0), -1)], dim=1)

    def forward_user(
        self,
        hist_ids: torch.LongTensor,    # [batch, max_hist_len]
        wish_ids: torch.LongTensor,    # [batch, max_hist_len]
        hist_dense: torch.FloatTensor, # [batch, max_hist_len, 3]
        wish_dense: torch.FloatTensor  # [batch, max_hist_len, 3]
    ):
        B = hist_ids.size(0)
        # embed & flatten history
        hist_emb = self.embed_sparse(hist_ids.view(-1)).view(B, self.max_hist_len, -1)
        wish_emb = self.embed_sparse(wish_ids.view(-1)).view(B, self.max_hist_len, -1)
        # flatten everything
        x = torch.cat([
            hist_emb.view(B, -1),
            wish_emb.view(B, -1),
            hist_dense.view(B, -1),
            wish_dense.view(B, -1),
        ], dim=1)
        return self.user_mlp(x)

    def forward_item(
        self,
        cand_ids: torch.LongTensor,     # [batch]
        cand_dense: torch.FloatTensor   # [batch,3]
    ):
        emb = self.embed_sparse(cand_ids)        # [batch,7*embed_dim]
        x   = torch.cat([emb, cand_dense], dim=1) # [batch,7*embed_dim+3]
        return self.item_mlp(x)

    def forward(
        self,
        hist_ids, wish_ids,
        hist_dense, wish_dense,
        cand_ids, cand_dense
    ):
        u = self.forward_user(hist_ids, wish_ids, hist_dense, wish_dense)
        i = self.forward_item(cand_ids, cand_dense)
        # final score by dot product
        return (u * i).sum(dim=1, keepdim=True)
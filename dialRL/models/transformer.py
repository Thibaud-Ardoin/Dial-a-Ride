from icecream import ic

import torch
import torch.nn as nn

from dialRL.utils import get_device

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, positions=None):
        N, seq_length, _ = x.shape
        if positions is None:
            positions = torch.tensor([0 for i in range(seq_length-1)] + [1]).expand(N, seq_length).to(self.device)

        out = self.dropout(
            (x.to(self.device) + positions.to(self.device)) #self.position_embedding(positions.to(self.device)))
        )


        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads=heads)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask, positions=None):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        if positions is None:
            positions = torch.tensor([0 for i in range(seq_length-1)] + [1]).expand(N, seq_length).to(self.device)

        x = self.dropout(self.word_embedding(x.to(self.device)) + self.position_embedding(positions.to(self.device)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        out = self.fc_out(x)
        return out


class Trans1(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx=0,
        trg_pad_idx=0,
        embed_size=128,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="",
        max_length=100,
    ):

        super(Trans1, self).__init__()
        self.device = device #get_device()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            self.device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            self.device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.embed_size = embed_size
        mapping_size = embed_size // 2
        scale = 10
        self.B_gauss = torch.normal(0, 1, size=(mapping_size, 2)) * scale


    def make_src_mask(self, src):
        # Little change towards original: src got embedding dimention in additon
        src_mask = (src[:,:,0] != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask.to(self.device)

    def forward(self, src, trg, positions=None):
        # Encoode positoin and env
        src = self.env_encoding(src)
        if not positions is None :
            positions = self.positional_encoding(positions)

        src_mask = self.make_src_mask(src)

        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask, positions=positions)
        out = self.decoder(trg, enc_src, src_mask, trg_mask, positions=positions)
        return out

    def env_encoding(self, src):
        w, ts, ds = src
        embeddig_size = self.embed_size #max(len(w), len(ts[0]), len(ds[0]))
        bsz = w[0].shape[-1]
        absant_vector = torch.zeros(bsz, dtype=torch.float64)
        w = w + [absant_vector] * (embeddig_size - len(w))
        world_emb = torch.stack(w)
        targets_emb = [torch.stack(target + [absant_vector] * (embeddig_size - len(target))) for target in ts]
        drivers_emb = [torch.stack(driver + [absant_vector] * (embeddig_size - len(driver))) for driver in ds]

        final_emb = torch.stack([world_emb] + targets_emb + drivers_emb)

        final_emb = final_emb.permute(2, 0, 1)
        return final_emb
        # Goal vector:
        #   (1 + |ts| + |ds|) x (max(4, len(ts[0]), len(ds[0]))) x bsz


    def fourier_feature(self, coordonates):
        coordonates = torch.tensor(coordonates)
        pi = torch.tensor(torch.acos(torch.zeros(1)).item() * 2 * 2)
        x = (pi * coordonates).double()
        transB = torch.transpose(self.B_gauss, 0, 1).double()
        if x.shape[1] == 4:
            transB = torch.cat([transB, transB])
        x_proj = x.matmul(transB)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)


    def positional_encoding(self, position):
        depot = self.fourier_feature(position[0])
        targets = [self.fourier_feature(pos) for pos in position[1]]
        drivers = [self.fourier_feature(pos) for pos in position[2]]

        d1 = torch.stack([depot])
        for target in targets :
            d2 = torch.stack([target])
            d1 = torch.cat([d1, d2])
        for driver in drivers :
            d3 = torch.stack([driver])
            d1 = torch.cat([d1, d3])

        return d1.permute(1, 0, 2)
        # return d1.permute(0, 2, 1)


if __name__ == "__main__":
    device = get_device()
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 5, 6, 4, 3, 9, 5, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).to(
        device
    )
    trg = torch.tensor([[0], [0]]).to(device)

    positions = [
                torch.tensor([[0.1,0.1],[0.1,0.1]]),
                torch.tensor([[[0.2, 0.2]], [[0.2, 0.2]]]),
                torch.tensor([[[0.3, 0.3, 0.4, 0.4], [1.1, 1.1, 1.2, 1.2]], [[0.3, 0.3, 0.4, 0.4], [1.1, 1.1, 1.2, 1.2]]])
                ]


    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Trans1(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(
        device
    )
    out = model(x, trg[:, :-1], positions=positions)
    print(out)
    print(out.argmax(-1))
    print(out.shape)

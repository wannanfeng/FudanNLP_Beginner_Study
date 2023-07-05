import torch
import torch.nn as nn

# self attention
class selfattention(nn.Module):
    def __init__(self, embed_size, heads):
        super(selfattention,self).__init__()
        self.emded_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), 'warning'

        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(self.heads*self.head_dim, embed_size)

    def forward(self, value, key, query, mask):
        N = query.shape[0]   # get the number of training example
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        # split embedding into self.heads pieces
        values = value.reshape(N, value_len, self.heads, self.head_dim)  # (n,v,h,d)
        keys = key.reshape(N, key_len, self.heads, self.head_dim)  # (n,k,h,d)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)  # (n,q,h,d)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)  # (n,h,q,k)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        attention = torch.softmax(energy / (self.emded_size ** (1/2)), dim=3)  # (n,h,q,k)   attention score
        out = torch.einsum('nhql,nlhd->nqhd', attention, values).reshape(N, query_len, self.heads*self.head_dim)
        return self.fc_out(out)

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock,self).__init__()
        self.attention = selfattention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

# Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_len_sentence):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.postion_ebedding = nn.Embedding(num_embeddings=max_len_sentence, embedding_dim=embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.postion_ebedding(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out

# Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, device):
        super(DecoderBlock,self).__init__()
        self.embed = embed_size
        self.heads = heads
        self.attention = selfattention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformerblock = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, source_mask, target_mask):
        attention = self.attention(x, x, x, target_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformerblock(value, key, query, source_mask)
        return out

# Decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_len):
        super(Decoder, self).__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_size, heads, dropout, forward_expansion, device
                )
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)
    def forward(self, x, encoder_out, source_mask, target_mask):
        N, seq_len = x.shape
        position = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(position))

        for layer in self.layers:
            x = layer(x, encoder_out, encoder_out, source_mask, target_mask)

        out = self.fc_out(x)
        return out

# final model
class Transformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, src_pad_idx, trg_pad_idx, embed_size=256, num_layers=6,
                 forward_expansion=4, heads=8, dropout=0, device='cuda', max_len=100):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            source_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_len
        )
        self.decoder = Decoder(
            target_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_len
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # (N ,1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        encoder_src = self.encoder(src, src_mask)
        out = self.decoder(trg, encoder_src, src_mask, trg_mask)

        return out

# test
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)
    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)

    out = model(x, trg[:, :-1])
    print(out.shape)
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from rezero.transformer import RZTXDecoderLayer, RZTXEncoderLayer
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class SketchModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, nclass, dropout=0.5):
        super().__init__()

        self.model_type = 'Transformer'
        self.src_mask = None
        encoder_layers = RZTXEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.value_embed = nn.Embedding(ntoken + 1, ninp)
        self.pos_embed = nn.Embedding(256, ninp)
        self.class_embed = nn.Embedding(nclass, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.value_embed.weight.data.uniform_(-initrange, initrange)
        self.pos_embed.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        v, p, c = src[..., 0], src[..., 1], src[..., 2]
        v = self.value_embed(v)
        p = self.pos_embed(p)
        c = self.class_embed(c)
        src = (v + p + c) * math.sqrt(self.ninp)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


def create_model(args):
    device = torch.device(args.device)
    model = SketchModel(dropout=args.dropout_rate,
                        nclass=args.num_sketch_classes,
                        nhead=args.num_att_heads,
                        nhid=4 * args.embed_dim,
                        ninp=args.embed_dim,
                        nlayers=args.num_transformer_layers,
                        ntoken=args.vocab_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    return model, device, optimizer

"""Generate random sample script."""

import math
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import dataset
from model import create_model


def nucleus_sampling(logit, mode, p=0.8):
    if mode == 0:
        logit[256:] = -float('inf')
    if mode == 1:
        logit[:256] = -float('inf')
        logit[512:] = -float('inf')
    if mode == 2:
        logit[256:512] = -float('inf')
        logit[513:] = -float('inf')
    if mode == 3:
        logit[256:513] = -float('inf')
    pr = torch.nn.functional.softmax(logit, -1)
    q1 = torch.sort(pr)[0]
    qq = torch.cumsum(q1, -1)
    t = torch.max(torch.where(qq < (1. - p), q1, torch.zeros_like(q1)))
    pr = torch.where(pr < t.view(-1), torch.zeros_like(pr), pr)
    return torch.distributions.categorical.Categorical(probs=pr).sample()


def generate(model, device, category):
    tokens, pos, mode = [[514, 0]], 0, 0
    model.eval()

    c = [[category]]
    c = torch.tensor(c, dtype=torch.long)[:, None, :].to(device)
    c = c[..., 0]
    c = model.class_embed(c)

    for i in range(256):
        src = torch.tensor(tokens, dtype=torch.long)[:, None, :].to(device)
        v, p = src[..., 0], src[..., 1]
        v = model.value_embed(v)
        p = model.pos_embed(p)
        s = v + p + c
        mask = model._generate_square_subsequent_mask(len(src)).to(device)
        model.src_mask = mask

        o = model.transformer_encoder(s, mask)
        o = model.decoder(o[-1])
        idx = nucleus_sampling(o.view(-1), mode)
        idx = idx.view(-1).item()
        if idx == 513:
            break
        if idx < 256:
            mode = 1
        if 256 <= idx < 512:
            mode = 2

        if idx == 512:
            mode = 3

        if idx < 256:
            pos = pos + 1
        tokens.append([idx, pos])

    model.train()

    t, p = zip(*tokens)
    return t


def plot_encoded_figure(fig, category, output=None):
    plt.close('all')
    x, y = [], []
    plt.figure(figsize=(3, 3))
    for token in fig:
        if token < 256:
            x.append(token)
        elif token < 512:
            y.append(512 - token)
        elif token == 512:
            if len(x) > 1:
                plt.plot(x, y, c='black', linewidth=1.)
            else:
                plt.scatter(x, y, s=1, c='black')
            x, y = [], []
        elif token == 513:
            break

    plt.xlim(0, 255)
    plt.ylim(0, 255)
    category = dataset.categories[category]
    plt.title(category)

    if output is not None:
        plt.savefig(output)


def main():
    parser = ArgumentParser()
    parser.add_argument('--category', default='cat', type=str)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--output-file', default='sample.png', type=Path)
    args = parser.parse_args()

    print('loading', args.checkpoint)
    dic = torch.load(args.checkpoint, map_location='cpu')
    dic['args'].device = args.device
    model, device, _ = create_model(dic['args'])
    cat = dataset.categories.index(args.category)
    model.load_state_dict(dic['model_state_dict'])
    with torch.no_grad():
        tokens = generate(model, device, cat)
    plot_encoded_figure(tokens, cat, args.output_file)


if __name__ == '__main__':
    main()

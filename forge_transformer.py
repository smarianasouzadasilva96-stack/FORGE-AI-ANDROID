"""Independent byte-level causal Transformer. No pretrained weights, APIs, or downloads.
Train: python training/forge_transformer.py train --data training/seed_corpus.txt
Sample: python training/forge_transformer.py sample --prompt 'def main'
Requires PyTorch installed separately. Checkpoint is for Python, NOT the Android app.
"""
import argparse
import math
import os
import random
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class CausalBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width), nn.Dropout(dropout))

    def forward(self, x):
        t = x.size(1)
        mask = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, attn_mask=mask, need_weights=False)
        x = x + y
        return x + self.ff(self.norm2(x))


class ForgeTransformer(nn.Module):
    def __init__(self, width=128, layers=3, heads=4, context=128, dropout=0.1):
        super().__init__()
        if width % heads or context < 8:
            raise ValueError('Invalid width/heads/context')
        self.config = dict(width=width, layers=layers, heads=heads, context=context, dropout=dropout)
        self.token = nn.Embedding(256, width)
        self.position = nn.Embedding(context, width)
        self.blocks = nn.ModuleList(CausalBlock(width, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 256, bias=False)
        # Untied output projection avoids very large self-token logits at initialization.
        nn.init.normal_(self.token.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

    def forward(self, ids, targets=None):
        b, t = ids.shape
        if t > self.config['context']:
            raise ValueError('Context exceeded')
        x = self.token(ids) + self.position(torch.arange(t, device=ids.device))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))
        loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1)) if targets is not None else None
        return logits, loss


def batch(data, context, size, device):
    starts = torch.randint(0, len(data) - context, (size,))
    x = torch.stack([data[i:i + context] for i in starts]).to(device)
    y = torch.stack([data[i + 1:i + context + 1] for i in starts]).to(device)
    return x, y


def train(args):
    if args.width <= 0 or args.layers <= 0 or args.heads <= 0 or args.context < 8 or args.width % args.heads:
        raise ValueError("Invalid model dimensions")
    if args.lr <= 0:
        raise ValueError("Learning rate must be positive")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    raw = Path(args.data).read_bytes()
    if len(raw) < args.context * 12:
        raise ValueError('Corpus too small: supply at least 12 * context bytes')
    data = torch.tensor(list(raw), dtype=torch.long)
    # Ensure both partitions can supply at least one complete training window.
    val_len = max(args.context + 2, int(len(data) * 0.1))
    if len(data) - val_len <= args.context + 1:
        raise ValueError('Corpus too small for independent train/validation windows')
    train_data, val_data = data[:-val_len], data[-val_len:]
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    model = ForgeTransformer(args.width, args.layers, args.heads, args.context).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_step = 0
    best = float('inf')
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        if checkpoint['config'] != model.config:
            raise ValueError('Checkpoint architecture does not match current arguments')
        model.load_state_dict(checkpoint['state'])
        if 'optimizer' in checkpoint:
            optim.load_state_dict(checkpoint['optimizer'])
        start_step = int(checkpoint['step'])
        best = float(checkpoint['validation_loss'])
    print(f'device={device} parameters={sum(p.numel() for p in model.parameters()):,} train_bytes={len(train_data)} validation_bytes={len(val_data)}')
    for step in range(start_step + 1, start_step + args.steps + 1):
        model.train()
        x, y = batch(train_data, args.context, args.batch, device)
        _, loss = model(x, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                scores = []
                for _ in range(args.eval_batches):
                    vx, vy = batch(val_data, args.context, args.batch, device)
                    scores.append(model(vx, vy)[1].item())
            score = sum(scores) / len(scores)
            print(f'step={step} train_loss={loss.item():.4f} validation_loss={score:.4f}')
            if score < best:
                best = score
                dest = Path(args.output)
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix('.tmp')
                torch.save({'config': model.config, 'state': model.state_dict(), 'optimizer': optim.state_dict(), 'step': step, 'validation_loss': score}, tmp)
                os.replace(tmp, dest)
    print(f'best_validation_loss={best:.4f} checkpoint={args.output}')


@torch.no_grad()
def sample(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = ForgeTransformer(**checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['state'])
    model.eval()
    ids = torch.tensor([list(args.prompt.encode('utf-8')) or [32]], dtype=torch.long, device=device)
    for _ in range(args.tokens):
        logits, _ = model(ids[:, -model.config['context']:])
        values = logits[:, -1] / args.temperature
        if args.top_k:
            threshold = values.topk(min(args.top_k, 256)).values[:, -1, None]
            values = values.masked_fill(values < threshold, float('-inf'))
        next_id = torch.multinomial(F.softmax(values, dim=-1), 1)
        ids = torch.cat((ids, next_id), dim=1)
    print(bytes(ids[0].tolist()).decode('utf-8', errors='replace'))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--data', required=True)
    t.add_argument('--output', default='training/checkpoints/forge-transformer.pt')
    t.add_argument('--width', type=int, default=128)
    t.add_argument('--layers', type=int, default=3)
    t.add_argument('--heads', type=int, default=4)
    t.add_argument('--context', type=int, default=128)
    t.add_argument('--steps', type=int, default=100)
    t.add_argument('--batch', type=int, default=8)
    t.add_argument('--lr', type=float, default=0.0003)
    t.add_argument('--eval-every', type=int, default=25)
    t.add_argument('--eval-batches', type=int, default=3)
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--cpu', action='store_true')
    t.add_argument('--resume', default=None, help='Resume from a matching checkpoint')
    s = sub.add_parser('sample')
    s.add_argument('--checkpoint', default='training/checkpoints/forge-transformer.pt')
    s.add_argument('--prompt', default='def main')
    s.add_argument('--tokens', type=int, default=100)
    s.add_argument('--temperature', type=float, default=0.8)
    s.add_argument('--top-k', type=int, default=40)
    s.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    if args.command == 'train':
        if args.steps < 1 or args.eval_every < 1 or args.batch < 1 or args.eval_batches < 1:
            parser.error('steps, eval-every, batch and eval-batches must be positive')
        train(args)
    else:
        if args.temperature <= 0 or args.tokens < 0 or args.top_k < 0:
            parser.error('temperature must be positive; tokens and top-k nonnegative')
        sample(args)


if __name__ == '__main__':
    main()

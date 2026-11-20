"""Measure held-out next-byte cross entropy and perplexity; no quality claims."""
import argparse
import json
import math
from pathlib import Path
import torch
from forge_transformer import ForgeTransformer

def evaluate(checkpoint, corpus, context, batch_size=8, batches=12, cpu=False):
    if batches < 1 or batch_size < 1: raise ValueError('Positive batch and batches required')
    device='cpu' if cpu or not torch.cuda.is_available() else 'cuda'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=ForgeTransformer(**ck['config']).to(device)
    model.load_state_dict(ck['state']); model.eval()
    data=torch.tensor(list(Path(corpus).read_bytes()),dtype=torch.long)
    context=min(context,model.config['context'])
    if len(data) <= context+1: raise ValueError('Evaluation corpus too small')
    generator=torch.Generator().manual_seed(2026)
    scores=[]
    with torch.inference_mode():
        for _ in range(batches):
            starts=torch.randint(0,len(data)-context-1,(batch_size,),generator=generator)
            x=torch.stack([data[i:i+context] for i in starts]).to(device)
            y=torch.stack([data[i+1:i+context+1] for i in starts]).to(device)
            _,loss=model(x,y); scores.append(loss.item())
    mean=sum(scores)/len(scores)
    return {'cross_entropy':round(mean,5),'perplexity':round(math.exp(mean),4),'samples':batches*batch_size,'context':context,'note':'Next-byte prediction only; not proof of conversational or coding competence'}

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--context',type=int,default=64); ap.add_argument('--batch',type=int,default=8); ap.add_argument('--batches',type=int,default=12); ap.add_argument('--cpu',action='store_true')
    a=ap.parse_args(); print(json.dumps(evaluate(a.checkpoint,a.data,a.context,a.batch,a.batches,a.cpu),indent=2))

"""Train FORGE's tiny 4-character model from scratch; exports Kotlin-compatible weights.
This is a character predictor, not a conversational assistant.
"""
import argparse, struct
from pathlib import Path
import numpy as np

ALPHABET = [chr(i) for i in range(32,127)] + ['\n','\t','á','é','í','ó','ú','ã','õ','ç','ê','â']
VOCAB=len(ALPHABET); WIDTH=48; WINDOW=4
MAPPING={c:i for i,c in enumerate(ALPHABET)}

def run(corpus, output, epochs=8, seed=42):
    text=Path(corpus).read_text(encoding='utf-8')
    if not 100 <= len(text) <= 200000: raise ValueError('Corpus length must be 100..200000 characters')
    rng=np.random.default_rng(seed)
    emb=((rng.random((VOCAB,WIDTH))-0.5)*0.12).astype('float32')
    proj=((rng.random((WIDTH,VOCAB))-0.5)*0.12).astype('float32')
    bias=np.zeros(VOCAB,dtype='float32')
    ids=np.array([MAPPING.get(c,0) for c in text],dtype=np.int32)
    split=max(2,int(len(ids)*0.9))
    posweights=np.arange(1,5,dtype=np.float32)/10
    def sample(i,update):
        ix=np.array([ids[i-4+k] if i-4+k>=0 else MAPPING[' '] for k in range(4)])
        hidden=np.tanh((emb[ix]*posweights[:,None]).sum(axis=0)).astype('float32')
        logits=hidden@proj+bias
        ex=np.exp(logits-logits.max()); prob=ex/ex.sum()
        loss=-np.log(max(float(prob[ids[i]]),1e-12))
        if update:
            delta=prob.copy();delta[ids[i]]-=1
            dh=(proj@delta)*(1-hidden*hidden)
            proj[:]-=0.012*np.outer(hidden,delta)
            bias[:]-=0.012*np.clip(delta,-1,1)
            for k in range(4): emb[ix[k]]-=0.012*dh*posweights[k]
        return loss
    for epoch in range(epochs):
        train_loss=np.mean([sample(i,True) for i in range(1,split)])
        valid_loss=np.mean([sample(i,False) for i in range(split,len(ids))])
        print(f'epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f} val_loss={valid_loss:.4f}')
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('wb') as f:
        f.write(struct.pack('>4i',0x464F5232,VOCAB,WIDTH,WINDOW))
        for matrix in (emb,proj):
            f.write(matrix.astype('>f4').tobytes(order='C'))
        f.write(bias.astype('>f4').tobytes())
    print('checkpoint:',out,'bytes:',out.stat().st_size)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--corpus',default='training/seed_corpus.txt');ap.add_argument('--output',default='app/src/main/assets/forge-seed-weights.bin');ap.add_argument('--epochs',type=int,default=8)
    args=ap.parse_args();run(args.corpus,args.output,args.epochs)

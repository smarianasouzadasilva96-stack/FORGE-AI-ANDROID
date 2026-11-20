import tempfile
import unittest
from pathlib import Path
from prepare_corpus import build
from forge_transformer import ForgeTransformer, train
from types import SimpleNamespace
import torch

class ForgeTests(unittest.TestCase):
    def test_corpus_dedup(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); (root/'a.py').write_text('print("ok")'); (root/'b.py').write_text('print("ok")'); (root/'c.txt').write_text('hello')
            result=build(root,root/'out.corpus'); self.assertEqual(result['files'],2)
    def test_forward_backward(self):
        model=ForgeTransformer(width=32,layers=1,heads=4,context=16,dropout=0.0)
        x=torch.randint(0,256,(2,16)); y=torch.randint(0,256,(2,16))
        logits,loss=model(x,y); self.assertEqual(tuple(logits.shape),(2,16,256))
        self.assertTrue(torch.isfinite(loss)); loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))
    def test_training_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            corpus=root/'corpus.txt'
            corpus.write_bytes((b'function add(a,b){return a+b;}\n' * 35))
            path=root/'model.pt'
            base=dict(data=str(corpus), output=str(path), width=16, layers=1, heads=4,
                      context=16, steps=1, batch=2, lr=0.001, eval_every=1,
                      eval_batches=1, seed=7, cpu=True, resume=None)
            train(SimpleNamespace(**base))
            self.assertTrue(path.exists())
            first=torch.load(path,weights_only=True)
            self.assertEqual(first['step'],1)
            self.assertIn('optimizer',first)
            base['resume']=str(path)
            train(SimpleNamespace(**base))

    def test_initial_loss_is_finite_and_reasonable(self):
        model=ForgeTransformer(width=32,layers=1,heads=4,context=16,dropout=0.0).eval()
        x=torch.randint(0,256,(2,16)); y=torch.randint(0,256,(2,16))
        with torch.no_grad():
            _,loss=model(x,y)
        self.assertGreater(loss.item(),4.0)
        self.assertLess(loss.item(),8.0)

    def test_causal_mask(self):
        model=ForgeTransformer(width=32,layers=1,heads=4,context=16,dropout=0.0).eval()
        a=torch.randint(0,256,(1,16)); b=a.clone(); b[0,12:]=0
        with torch.no_grad():
            la,_=model(a); lb,_=model(b)
        self.assertTrue(torch.allclose(la[:,:12],lb[:,:12],atol=1e-5))

if __name__=='__main__': unittest.main()

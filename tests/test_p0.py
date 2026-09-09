import importlib.util,unittest
import torch
from torch import nn

class P0Test(unittest.TestCase):
 def test_only_active_heads_and_embedding_grad(self):
  self.assertIsNotNone(importlib.util.find_spec('encoder.pretrain_v2'),'P0 module required')
  from encoder.pretrain_v2 import P0Model
  encoder=nn.Sequential(nn.Flatten(),nn.Linear(3*16*16,512))
  model=P0Model(encoder,['depth'])
  self.assertEqual(set(model.decoders),{'depth'})
  x=torch.randn(2,3,16,16);targets={'depth':torch.randn(2,1,24,32)}
  loss,parts=model.loss(x,targets);loss.backward()
  self.assertTrue(torch.isfinite(loss));self.assertGreater(encoder[1].weight.grad.abs().sum().item(),0)
  self.assertEqual(set(parts),{'depth','total'})
if __name__=='__main__':unittest.main()

"""P0 heads exclusively consume the exported 512-dimensional embedding."""
import torch
from torch import nn
from torch.nn import functional as F

class DenseDecoder(nn.Module):
 def __init__(self,channels):
  super().__init__()
  self.fc=nn.Linear(512,64*8*10)
  self.net=nn.Sequential(nn.ConvTranspose2d(64,32,4,2,1),nn.GELU(),nn.ConvTranspose2d(32,16,4,2,1),nn.GELU(),nn.ConvTranspose2d(16,8,4,2,1),nn.GELU(),nn.Conv2d(8,channels,3,padding=1))
 def forward(self,z):return self.net(self.fc(z).reshape(-1,64,8,10))

class P0Model(nn.Module):
 def __init__(self,encoder,active_heads,marker_count=1200,dynamic_weight=None):
  super().__init__();self.encoder=encoder;self.decoders=nn.ModuleDict();self.dynamic_weight=dynamic_weight
  for name in active_heads:
   if name=='marker':self.decoders[name]=nn.Sequential(nn.Linear(512,512),nn.GELU(),nn.Linear(512,marker_count*2),nn.Unflatten(1,(marker_count,2)))
   elif name in ('depth','rgb'):self.decoders[name]=DenseDecoder(1 if name=='depth' else 3)
   else:raise ValueError(name)
  if not self.decoders:raise ValueError('At least one active head required')
  if dynamic_weight is not None:
   if dynamic_weight<0:raise ValueError('dynamic_weight must be non-negative')
   self.dynamic_decoder=nn.Sequential(nn.Linear(256,512),nn.GELU(),nn.Linear(512,marker_count*2),nn.Unflatten(1,(marker_count,2)))
 def loss(self,x,targets):
  if self.dynamic_weight is None:z=self.encoder(x);detail_dynamic=None
  else:z,detail_dynamic=self.encoder.forward_with_details(x)
  parts={}
  for name,decoder in self.decoders.items():
   pred=decoder(z);target=targets[name]
   if pred.ndim==4:pred=F.interpolate(pred,size=target.shape[-2:],mode='bilinear',align_corners=False)
   valid=torch.isfinite(target)
   if not valid.any():raise ValueError(f'No valid {name} target')
   parts[name]=(pred[valid]-target[valid]).square().mean()
  total=sum(parts.values())
  if self.dynamic_weight is not None:
   pred=self.dynamic_decoder(detail_dynamic);target=targets['delta_marker'];valid=torch.isfinite(target)
   if not valid.any():raise ValueError('No valid delta_marker target')
   parts['delta_marker']=(pred[valid]-target[valid]).square().mean()
   total=total+self.dynamic_weight*parts['delta_marker']
  return total,{**parts,'total':total}

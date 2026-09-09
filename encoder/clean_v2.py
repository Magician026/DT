"""Trajectory splits and active-target-only clean GSmini loading (no simulator imports)."""
import hashlib
import json
from pathlib import Path
import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

PREPROCESS = {'sensor':'gsmini','field':'rgb_marker','channels':'native_cv2_decode_no_swap',
              'input_dtype':'uint8','scale':255.0,'image_size':[256,256],
              'resize':'torchvision_bilinear_antialias','normalization':'none'}

def manifest_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def make_manifest(root, seed=42):
    files=sorted(p.name for p in Path(root).glob('*.hdf5'))
    if len(files)<2: raise ValueError('Need >=2 independent trajectories')
    order=np.random.default_rng(seed).permutation(len(files))
    nval=max(1,round(len(files)*.1))
    return {'seed':seed,'grouping':'one_file_one_trajectory_both_sensors','train':sorted(files[i] for i in order[nval:]),'val':sorted(files[i] for i in order[:nval])}

def decode(value):
    x=cv2.imdecode(np.frombuffer(value,dtype=np.uint8),cv2.IMREAD_COLOR)
    if x is None: raise ValueError('Corrupt tactile JPEG')
    return x

class CleanDataset(Dataset):
    def __init__(self,root,files,active_heads,normalization,image_size=(256,256),stride=1):
        self.root=Path(root); self.active_heads=tuple(active_heads);self.normalization=normalization
        self.image_size=tuple(image_size);self.index=[]
        if not set(active_heads)<= {'depth','marker','rgb'}: raise ValueError(active_heads)
        if stride<1: raise ValueError('stride must be positive')
        for name in files:
            with h5py.File(self.root/name,'r') as f:
                for side in ('left','right'):
                    key=f'tactile/{side}_gsmini';s=f[key];n=len(s['rgb_marker'])
                    for head in active_heads:
                        if len(s[head])!=n:raise ValueError(f'Length mismatch {name}/{key}/{head}')
                    self.index.extend((name,key,i) for i in range(0,n,stride))
    def __len__(self):return len(self.index)
    def __getitem__(self,i):
        name,key,frame=self.index[i]
        with h5py.File(self.root/name,'r') as f:
            s=f[key];x=TF.resize(TF.to_tensor(decode(s['rgb_marker'][frame])),self.image_size,antialias=True)
            targets={}
            for head in self.active_heads:
                raw=s[head][frame]
                if head=='depth':raw=np.asarray(raw,np.float32)[None]
                elif head=='marker':raw=np.asarray(raw[1]-raw[0],np.float32)
                else:raw=TF.to_tensor(decode(raw)).numpy()
                if not np.isfinite(raw).all():raise ValueError(f'Nonfinite target {name}/{key}/{frame}/{head}')
                st=self.normalization[head]
                target=torch.from_numpy((raw-np.asarray(st['mean'],np.float32))/np.asarray(st['std'],np.float32))
                targets[head]=target
        if not torch.isfinite(x).all():raise ValueError('nonfinite image')
        return x,targets

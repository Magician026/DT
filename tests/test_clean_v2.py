import importlib.util
import tempfile
from pathlib import Path
import unittest
import cv2
import h5py
import numpy as np
import torch

class CleanContractTest(unittest.TestCase):
    def test_active_only_and_group_split(self):
        self.assertIsNotNone(importlib.util.find_spec('encoder.clean_v2'), 'new trajectory loader required')
        from encoder.clean_v2 import make_manifest, CleanDataset
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for i in range(10):
                with h5py.File(root/f'{i}.hdf5','w') as f:
                    for side in ('left','right'):
                        s=f.create_group(f'tactile/{side}_gsmini')
                        raw=np.zeros((24,32,3),np.uint8);raw[...,0]=230
                        encoded=cv2.imencode('.jpg',raw)[1].tobytes()
                        s.create_dataset('rgb_marker',data=np.array([encoded]*2,dtype=f'S{len(encoded)}'))
                        s.create_dataset('depth',data=np.full((2,24,32),4,np.float32))
            manifest=make_manifest(root,42)
            self.assertEqual((len(manifest['train']),len(manifest['val'])),(9,1))
            self.assertFalse(set(manifest['train'])&set(manifest['val']))
            ds=CleanDataset(root,manifest['train'],active_heads=['depth'],normalization={'depth':{'mean':2.,'std':2.}},image_size=(24,32))
            x,t=ds[0]
            self.assertEqual(len(ds),36)
            self.assertEqual(set(t),{'depth'})
            self.assertEqual(tuple(x.shape),(3,24,32))
            self.assertGreater(float(x[0].mean()),.8)
            self.assertTrue(torch.allclose(t['depth'],torch.ones_like(t['depth'])))
            self.assertEqual(make_manifest(root,42),manifest)

if __name__=='__main__': unittest.main()

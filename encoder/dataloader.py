import torch
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('..')
from envs.utils.transforms import *
from envs.utils.data import HDF5Handler

import time
import h5py
import numpy as np
from tqdm import tqdm
import threading

_worker_hdf5_cache = {}

PRISM_NAMES = [
    'CircleShell', 'Cross', 'Cubehole', 'Cuboid', 'Cylinder',
    'Doubleslope', 'Hemisphere', 'Line', 'Pacman', 'S', 'Sphere',
    'Star', 'Tetrahedron', 'Torus'
]

DATASET_SCHEMAS = {
    'legacy_contact_gs': {
        'left_sensor': 'tactile/left_tactile',
        'right_sensor': 'tactile/right_tactile',
        'pose_candidates': ('actor/prism',),
        'image_size': (320, 240),
    },
    'gsmini': {
        'left_sensor': 'tactile/left_gsmini',
        'right_sensor': 'tactile/right_gsmini',
        'pose_candidates': ('actor/prism', 'actor/bottle'),
        # Decoder outputs are 256x256; use the same target size for the new
        # schema so RGB/depth losses have matching shapes.
        'image_size': (256, 256),
    },
}


def discover_hdf5_paths(schema='legacy_contact_gs', data_root=None):
    """Discover encoder files without changing the legacy default layout."""
    if schema not in DATASET_SCHEMAS:
        raise ValueError(
            f'Unknown encoder dataset schema {schema!r}; '
            f'expected one of {tuple(DATASET_SCHEMAS)}'
        )

    if schema == 'legacy_contact_gs':
        root = Path(data_root) if data_root is not None else Path('../data/contact-gs')
        hdf5_paths = []
        for name in PRISM_NAMES:
            hdf5_paths.extend(sorted((root / name / 'hdf5').glob('*.hdf5')))
    else:
        root = Path(data_root) if data_root is not None else Path('../data/insert_HDMI/clean')
        hdf5_paths = sorted(root.glob('*.hdf5'))

    if not hdf5_paths:
        raise FileNotFoundError(
            f'No HDF5 files found for schema={schema!r} under {root}. '
            'Check --schema and --data_root.'
        )
    return hdf5_paths

def worker_init_fn(worker_id):
    _worker_hdf5_cache[threading.get_ident()] = {}

def worker_clear_fn():
    """清理当前线程的 HDF5 文件"""
    ident = threading.get_ident()
    if ident in _worker_hdf5_cache:
        for f in _worker_hdf5_cache[ident].values():
            f.close()
        del _worker_hdf5_cache[ident]

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_paths:list, schema='legacy_contact_gs', image_size=None):
        self.handler = HDF5Handler()

        if schema not in DATASET_SCHEMAS:
            raise ValueError(
                f'Unknown encoder dataset schema {schema!r}; '
                f'expected one of {tuple(DATASET_SCHEMAS)}'
            )
        if not hdf5_paths:
            raise ValueError('HDF5Dataset requires at least one HDF5 file.')
        self.schema = schema
        schema_config = DATASET_SCHEMAS[schema]
        self.image_size = tuple(image_size or schema_config['image_size'])
        if len(self.image_size) != 2 or any(size <= 0 for size in self.image_size):
            raise ValueError(f'image_size must be two positive integers, got {self.image_size}')
        self.name = {}
        for lr_tag, sensor_path in (
            ('left', schema_config['left_sensor']),
            ('right', schema_config['right_sensor']),
        ):
            self.name.update({
                f'{lr_tag}_marked_rgb': f'{sensor_path}/rgb_marker',
                f'{lr_tag}_marker': f'{sensor_path}/marker',
                f'{lr_tag}_depth': f'{sensor_path}/depth',
                f'{lr_tag}_rgb': f'{sensor_path}/rgb',
                f'{lr_tag}_pose': f'{sensor_path}/pose',
            })
        self.name['prism_pose'] = self._resolve_pose_path(
            hdf5_paths[0], schema_config['pose_candidates']
        )
        
        self._data_metadata = []
        for hdf5_file in tqdm(hdf5_paths, desc='Loading HDF5 metadata'):
            hdf5_path = str(hdf5_file)
            hdf5_metadata = self.handler.load_hdf5_metadata(hdf5_path)
            for length in range(hdf5_metadata['length']):
                self._data_metadata.extend([{
                    'hdf5_path': hdf5_path,
                    'index': length,
                    'lr_tag': 'left'
                }, {
                    'hdf5_path': hdf5_path,
                    'index': length,
                    'lr_tag': 'right'
                }])

        self.trans = transforms.Compose([
            transforms.ToTensor(), # auto convert HWC to CHW and scale to [0, 1]
            transforms.Resize(self.image_size),
        ])
        self.depth_trans = transforms.Resize(self.image_size)

    def __len__(self): 
        return len(self._data_metadata)

    def __getitem__(self, idx):
        hdf5_path = self._data_metadata[idx]['hdf5_path']
        real_idx  = self._data_metadata[idx]['index']
        lr_tag    = self._data_metadata[idx]['lr_tag']

        thread_id = threading.get_ident()
        if thread_id not in _worker_hdf5_cache:
            _worker_hdf5_cache[thread_id] = {}
        cache = _worker_hdf5_cache[thread_id]

        if hdf5_path not in cache:
            cache[hdf5_path] = h5py.File(hdf5_path, 'r')

        f = cache[hdf5_path]

        data = {}
        for key, path in self.name.items():
            if 'rgb' in path:
                data[key] = self.handler.stream_to_img(
                    f[path][real_idx],
                    resize=False, convert_channels=False, path=path
                ).squeeze(0)
            else:
                data[key] = f[path][real_idx]
        
        rgb = self.trans(data[f'{lr_tag}_rgb'])
        marked_rgb = self.trans(data[f'{lr_tag}_marked_rgb'])
        marker = data[f'{lr_tag}_marker'][0, :63] / np.array([320, 240], dtype=np.float32)
        depth = torch.from_numpy(
            (data[f'{lr_tag}_depth'].astype(np.float32) - 24.0) / (34.0 - 24.0)
        ).unsqueeze(0)
        if tuple(depth.shape[-2:]) != self.image_size:
            depth = self.depth_trans(depth)
        pose = Pose.from_list(data[f'{lr_tag}_pose'])
        prism_pose = Pose.from_list(data['prism_pose'])
        vec = torch.tensor(prism_pose.rebase(pose).tolist())

        sample = {
            'pose': vec,
            'rgb': rgb,
            'marked_rgb': marked_rgb,
            'marker': torch.from_numpy(marker),
            'depth': depth,
        }
        return sample

    @staticmethod
    def _resolve_pose_path(hdf5_path, candidates):
        with h5py.File(hdf5_path, 'r') as f:
            for candidate in candidates:
                if candidate in f:
                    return candidate
        raise KeyError(
            f'None of pose candidates {tuple(candidates)} exists in {hdf5_path}'
        )

import os
from tqdm import tqdm
if __name__ == '__main__':
    prism_names = ['CircleShell', 'Cross', 'Cubehole', 'Cuboid', 'Cylinder', 'Doubleslope', 'Hemisphere', 'Line', 'Pacman', 'S', 'Sphere', 'Star', 'Tetrahedron', 'Torus']
    hdf5_paths = []
    for name in tqdm(prism_names, desc='Loading'):
        hdf5_paths.extend(list(Path(f'../data/demo2/{name}/hdf5').glob('*.hdf5')))
    print(f'Found {len(hdf5_paths)} hdf5 files.')
    
    os.system('mkdir -p /dev/shm/hdf5_cache')
    for name in prism_names:
        os.system(f'mkdir -p /dev/shm/hdf5_cache/{name}')
    for p in tqdm(hdf5_paths, desc='Copying to shm'):
        name = p.parent.parent.stem
        os.system(f'cp {str(p)} /dev/shm/hdf5_cache/{name}/{p.stem}.hdf5')

    data = HDF5Dataset(hdf5_paths)
    
    # depth = data[0]['marker']
    # print(depth.shape, depth.min(), depth.max())

    import time
    it = iter(data)
    for i in range(10):
        t0 = time.time()
        batch = next(it)
        print(f"Batch {i}: {time.time() - t0:.3f}s")

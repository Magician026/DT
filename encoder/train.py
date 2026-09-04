import json
import time
import random
import torch
import torch.nn as nn
from pathlib import Path

from torchvision import transforms
import matplotlib.pyplot as plt

from network import *
from dataloader import *
from torch.utils.data import random_split

data_length = 1000
backbone = 'resnet18'
encoder_type = 'original'
ckpt_root = 'ablation/data_1000'
weights = {}
timestr = time.strftime(r"%Y%m%d-%H%M%S", time.localtime())


def get_save_root():
    """Return an encoder-specific output directory without overwriting baseline runs."""
    if encoder_type == 'original':
        return Path(f'{ckpt_root}/{backbone}/{timestr}')
    return Path(f'{ckpt_root}/{encoder_type}/{backbone}/{timestr}')


def log(msg):
    save_path = get_save_root() / 'log.log'
    with open(save_path, 'a') as f:
        f.write(msg + '\n')

def train(
    train_loader:DataLoader,
    valid_loader:DataLoader,
    epoches=5,
    lr=1e-4,
    loss_weights:dict={'rgb_marked': 1.0},
    run_metadata:dict|None=None,
):
    global backbone, encoder_type, timestr, ckpt_root, weights
    save_root = get_save_root()
    save_root.mkdir(parents=True, exist_ok=True)
    run_config = {
        'status': 'config',
        'backbone': backbone,
        'encoder_type': encoder_type,
        'epoches': epoches,
        'lr': lr,
        'loss_weights': loss_weights,
    }
    if run_metadata:
        run_config.update(run_metadata)
    log(json.dumps(run_config, ensure_ascii=False))

    device = torch.device('cuda')
    supervise = list(loss_weights.keys())
    model = Tactile(
        backbone=backbone,
        supervise=supervise,
        encoder_type=encoder_type,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    valid_losses = []
    best_loss = 1e9
    
    for epoch in range(epoches):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        pbar.set_description(f'Epoch {epoch+1:4d}/{epoches:4d}, Training')
        for idx, d in pbar:
            x:torch.Tensor = d['marked_rgb'].to(device)
            y = {s: d[s].to(device) for s in supervise}
            outputs = model.reconstruct(x)

            loss, loss_dict = model.loss(outputs, y, weights=loss_weights)
            train_loss += loss_dict['total']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': loss_dict['total']})
            if idx % 20 == 0:
                log(json.dumps({
                    'status': 'train',
                    'epoch': epoch + 1,
                    'batch': idx + 1,
                    'loss': loss_dict['total'],
                    'losses': loss_dict
                }, ensure_ascii=False))
            
        train_losses.append(train_loss / max(len(train_loader), 1))

        valid_loss = 0.0
        model.eval()
        with torch.no_grad():
            pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), leave=True)
            pbar.set_description(f'Epoch {epoch+1:4d}/{epoches:4d}, Validating')
            for idx, d in pbar:
                x:torch.Tensor = d['marked_rgb'].to(device)
                y = {s: d[s].to(device) for s in supervise}
                outputs = model.reconstruct(x)

                loss, loss_dict = model.loss(outputs, y, weights=loss_weights)
                valid_loss += loss_dict['total']
                
                pbar.set_postfix({'loss': loss_dict['total']})
                if idx % 20 == 0:
                    log(json.dumps({
                        'status': 'eval',
                        'epoch': epoch + 1,
                        'batch': idx + 1,
                        'loss': loss_dict['total'],
                        'losses': loss_dict
                    }, ensure_ascii=False))
        
        torch.save(model.state_dict(), str(save_root / f'ep{epoch}.pth'))
        valid_mean_loss = valid_loss / max(len(valid_loader), 1)
        valid_losses.append(valid_mean_loss)
        if valid_mean_loss < best_loss:
            best_loss = valid_mean_loss
            torch.save(model.state_dict(), str(save_root / 'best.pth'))
            print(f'  Best model saved with recon loss {best_loss:.6f}')

def main(
    schema='legacy_contact_gs',
    data_root=None,
    image_size=None,
    epoches=5,
    batch_size=64,
    num_workers=8,
    seed=42,
):
    global data_length
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    hdf5_paths = discover_hdf5_paths(schema=schema, data_root=data_root)
    print(f'Found {len(hdf5_paths)} hdf5 files for schema={schema}.')
    data = HDF5Dataset(hdf5_paths, schema=schema, image_size=image_size)
    if data_length <= 1:
        raise ValueError('data_length must be at least 2 for a train/validation split.')
    if data_length > len(data):
        raise ValueError(
            f'data_length={data_length} exceeds available dataset length={len(data)}.'
        )
    data._data_metadata = list(np.array(data._data_metadata)[
        np.random.choice(len(data._data_metadata), data_length, replace=False)])
    print(f'Dataset length: {len(data)}')
    
    valid_size = int(len(data) * 0.2)
    generator = torch.Generator().manual_seed(seed)
    train_size = len(data) - valid_size
    train_data, valid_data = random_split(
        data, [train_size, valid_size], generator=generator)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        pin_memory=True
    )
    train(
        train_loader,
        valid_loader,
        epoches=epoches,
        lr=1e-3,
        loss_weights=weights,
        run_metadata={
            'dataset_schema': schema,
            'data_root': str(data_root) if data_root is not None else None,
            'image_size': list(data.image_size),
            'available_files': len(hdf5_paths),
            'sample_count': len(data),
            'batch_size': batch_size,
            'num_workers': num_workers,
            'seed': seed,
        },
    )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('config', type=str, default='all')
    parser.add_argument('data_num', type=int, default=1000)
    parser.add_argument(
        '--encoder_type',
        choices=('original', 'detail_v1'),
        default='original',
        help='Encoder implementation; original preserves the baseline path.',
    )
    parser.add_argument(
        '--schema',
        choices=tuple(DATASET_SCHEMAS),
        default='legacy_contact_gs',
        help='Encoder HDF5 schema; legacy default preserves the old data layout.',
    )
    parser.add_argument(
        '--data_root',
        type=Path,
        default=None,
        help='HDF5 root. For gsmini this is a directory containing *.hdf5 files.',
    )
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        default=None,
        metavar=('HEIGHT', 'WIDTH'),
        help='Override image/depth target size; gsmini defaults to 256 256.',
    )
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    backbone = 'resnet18'
    encoder_type = args.encoder_type
    if args.config == 'shape_pathway':
        ckpt_root = 'ablation/shape_pathway'
        weights = {
            'marked_rgb': 1.0,
            'rgb': 1.0,
            'depth': 0.0,
            'marker': 0.0,
            'pose': 0.0
        }
    elif args.config == 'contact_pathway':
        ckpt_root = 'ablation/contact_pathway'
        weights = {
            'marked_rgb': 0.0,
            'rgb': 0.0,
            'depth': 1.0,
            'marker': 1.0,
            'pose': 0.0
        }
    elif args.config == 'contact_shape':
        ckpt_root = 'ablation/contact_shape'
        weights = {
            'marked_rgb': 1.0,
            'rgb': 1.0,
            'depth': 0.5,
            'marker': 0.5,
            'pose': 0.0
        }
    else:
        ckpt_root = f'ablation/data_{args.data_num}'
        data_length = args.data_num
        weights = {
            'marked_rgb': 1.0,
            'rgb': 1.0,
            'depth': 0.5,
            'marker': 0.5,
            'pose': 0.5
        }
    main(
        schema=args.schema,
        data_root=args.data_root,
        image_size=tuple(args.image_size) if args.image_size is not None else None,
        epoches=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

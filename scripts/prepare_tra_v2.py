"""Strict Spatial warm-start and a bounded numerical check; no pretraining."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import torch
from torchvision import transforms
from torch.nn import functional as F

from encoder.clean_v2 import PREPROCESS
from encoder.detail_v2 import load_encoder_checkpoint, save_encoder_checkpoint
from encoder.tra_v2 import warm_start_tra_encoder

ANCHOR_SHA256 = '914d9d8315a8ccd7fd88ea853b20375a90010ac45968e88e9d2eb386abffe448'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spatial-checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    source, out = Path(args.spatial_checkpoint), Path(args.out)
    if sha(source) != ANCHOR_SHA256:
        raise RuntimeError('TRA requires the verified Spatial 31/100 pretrained encoder')
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    anchor_copy = out/'spatial_anchor.pt'
    shutil.copy2(source, anchor_copy)
    spatial, _ = load_encoder_checkpoint(anchor_copy, expected_encoder_type='detail_v2', expected_preprocess=PREPROCESS)
    tra, accounting = warm_start_tra_encoder(anchor_copy)
    spatial.eval(); tra.eval()
    for key, value in spatial.state_dict().items():
        torch.testing.assert_close(tra.state_dict()[key], value, rtol=0, atol=0)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256,256))])
    windows = []
    # Two real sensors at two anchors in a training episode. No future reads.
    sample_episode = Path(args.data)/'episode_27.hdf5'
    with h5py.File(sample_episode, 'r') as root:
        for side in ('tac_left', 'tac_right'):
            frames = root[f'/observations/images/{side}']
            for anchor in (0, 3):
                indices = [max(0, anchor-offset) for offset in (3,2,1,0)]
                windows.append(torch.stack([transform(frames[i]) for i in indices]))
    inputs = torch.stack(windows)
    with torch.no_grad():
        baseline = spatial(inputs[:,-1])
        semantic, details = tra.forward_spatial_details(inputs)
        raw_current = torch.cat((semantic, details[:,-1]), dim=-1)
        torch.testing.assert_close(raw_current, baseline, rtol=1e-5, atol=1e-5)
        alpha = tra.alpha.clone()
        tra.alpha.fill_(-20.)
        near_zero = tra(inputs)
        expected = torch.cat((baseline[:,:256], tra.fusion_norm(baseline[:,256:])), dim=-1)
        torch.testing.assert_close(near_zero, expected, rtol=1e-5, atol=1e-5)
        cosine = F.cosine_similarity(near_zero, baseline).tolist()
        relative_l2 = ((near_zero-baseline).norm(dim=-1)/baseline.norm(dim=-1)).tolist()
        # Final LayerNorm is explicitly new: quantify its effect, not exact equality.
        if min(cosine) < 0.85:
            raise RuntimeError(f'Warm-start final representation cosine too low: {cosine}')
        tra.alpha.copy_(alpha)
        initial = tra(inputs)
        initial_cosine = F.cosine_similarity(initial, baseline).tolist()
    checkpoint = out/'init.pt'
    save_encoder_checkpoint(checkpoint, tra, PREPROCESS)
    reloaded, _ = load_encoder_checkpoint(checkpoint, expected_encoder_type='detail_v2_tra', expected_preprocess=PREPROCESS)
    reloaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(reloaded(inputs), initial, rtol=0, atol=0)
    report = {
        'status': 'passed', 'sample_episode': str(sample_episode), 'sample_anchors': [0,3],
        'sensors': ['tac_left','tac_right'], 'source_checkpoint': str(source),
        'source_checkpoint_sha256': ANCHOR_SHA256, 'accounting': accounting,
        'all_spatial_state_exact': True,
        'raw_current_max_abs': float((raw_current-baseline).abs().max()),
        'near_zero_gate': float(torch.sigmoid(torch.tensor(-20.))),
        'near_zero_final_cosine': cosine, 'near_zero_final_relative_l2': relative_l2,
        'initial_gate': float(torch.sigmoid(tra.alpha.detach())),
        'initial_final_cosine': initial_cosine,
        'limitation': 'Final LayerNorm changes the Spatial post-GELU detail scale even at zero residual; all inherited weights and pre-fusion current features match.',
        'strict_reload': True, 'pretraining_optimizer_updates': 0,
    }
    (out/'warm_start_sanity.json').write_text(json.dumps(report, indent=2))
    (out/'completion.json').write_text(json.dumps({
        'status':'completed', 'stage':'spatial_warm_start_only', 'smoke':False,
        'checkpoint':'init.pt', 'checkpoint_sha256':sha(checkpoint),
        'source_commit':args.source_commit, 'anchor_sha256':ANCHOR_SHA256,
        'encoder_type':'detail_v2_tra', 'optimizer_updates':0,
    }, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

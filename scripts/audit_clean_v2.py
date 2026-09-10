"""Read-only dataset audit; fixed train-only normalization, portable manifests."""
import argparse,json
from pathlib import Path
import h5py,numpy as np
from encoder.clean_v2 import make_manifest,manifest_hash,decode,PREPROCESS

def validate_marker_pairs(m):
 # gen_marker_flow applies visibility/tracking/selection to both planes together.
 # Slot identity across frames is not invariant. Targets remain frame-local pairs.
 if m.ndim!=4 or m.shape[1:]!=(2,1200,2) or len(m)==0:raise ValueError('Invalid marker pair shape')
 if not np.isfinite(m).all():raise ValueError('Nonfinite marker pair')
 return {'reference_changes_across_frames':not bool(np.allclose(m[:,0],m[0,0],atol=1e-5)),
         'maximum_reference_slot_change':float(np.max(np.abs(m[:,0]-m[0,0]))),
         'semantics':'frame-local initial/current pairs; visibility filtering and repeat-last padding shared across planes'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--clean',required=True);p.add_argument('--act',required=True);p.add_argument('--out',required=True);p.add_argument('--depth-only',action='store_true');a=p.parse_args()
 root=Path(a.clean);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 manifest=make_manifest(root);(out/'split.json').write_text(json.dumps(manifest,indent=2))
 sums={k:[0,0.,0.] for k in (('depth',) if a.depth_only else ('depth','marker'))}; rows=[]
 for name in manifest['train']+manifest['val']:
  with h5py.File(root/name) as f:
   row={'file':name,'sensors':{}}
   for side in ('left','right'):
    s=f[f'tactile/{side}_gsmini'];n=len(s['rgb_marker']);m=None if a.depth_only else s['marker'][::10];d=s['depth'][::10]
    if (m is not None and not np.isfinite(m).all()) or not np.isfinite(d).all():raise ValueError(name)
    if m is not None:assert m.shape[1:]==(2,1200,2)
    assert d.shape[1:]==(240,320)
    marker_audit=None if m is None else validate_marker_pairs(m)
    image=decode(s['rgb_marker'][0]);assert image.shape==(240,320,3)
    delta=None if m is None else m[:,1]-m[:,0]
    row['sensors'][side]={'frames':n,'marker_pair_audit':marker_audit,'depth_range':[float(d.min()),float(d.max())],'marker_displacement_range':None if delta is None else [float(delta.min()),float(delta.max())],'image_shape':list(image.shape)}
    if name in manifest['train']:
     for key,v in ([('depth',d)] if a.depth_only else [('depth',d),('marker',delta)]):
      v=v.astype(np.float64);sums[key][0]+=v.size;sums[key][1]+=v.sum();sums[key][2]+=(v*v).sum()
   rows.append(row)
  print(name,flush=True)
 normalization={}
 for k,(n,s,ss) in sums.items():
  normalization[k]={'mean':s/n,'std':max(float(np.sqrt(ss/n-(s/n)**2)),1e-6),'count':n,'statistics_frame_stride':10,'source':'train_only'}
 act=Path(a.act);episodes=sorted(act.glob('episode_*.hdf5'));q=[];actions=[]
 for file in episodes:
  with h5py.File(file) as f:
   q.append(f['observations/qpos'][:]);actions.append(f['action'][:]);assert q[-1].shape==actions[-1].shape;assert q[-1].shape[1]==8
   assert np.isfinite(q[-1]).all() and np.isfinite(actions[-1]).all()
 with h5py.File(root/'0.hdf5') as c,h5py.File(act/'episode_0.hdf5') as f:
  image=decode(c['tactile/left_gsmini/rgb_marker'][0]);actual=f['observations/images/tac_left'][0]
  match={'clean0_act0_max_abs':int(np.abs(image.astype(int)-actual.astype(int)).max()),'clean0_act0_mean_abs':float(np.abs(image.astype(int)-actual.astype(int)).mean())}
 report={'active_heads':list(sums),'marker_exclusion_reason':'reference semantics unverified; target disabled by P0 rule B' if a.depth_only else None,'split_hash':manifest_hash(manifest),'files':len(rows),'total_sensor_frames':sum(s['frames'] for r in rows for s in r['sensors'].values()),'normalization':normalization,'preprocess':PREPROCESS,'trajectory_audit':rows,'policy':{'episodes':len(episodes),'total_frames':sum(len(x) for x in q),'qpos_range':[float(np.concatenate(q).min()),float(np.concatenate(q).max())],'action_range':[float(np.concatenate(actions).min()),float(np.concatenate(actions).max())],'image_correspondence_check':match,'split':'separate ACT protocol, no encoder/policy split equivalence claimed'}}
 (out/'data_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='trajectory_audit'},indent=2))
if __name__=='__main__':main()

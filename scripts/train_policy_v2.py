"""ACT integration with exact optimizer-update budget and validated dependencies."""
import argparse,fcntl,hashlib,json,os,pickle,random,sys,time
from pathlib import Path
import numpy as np
import torch,yaml
from torch.utils.data import DataLoader
if not __debug__:raise RuntimeError('Validation requires Python without -O')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'policy/ACT'))
from act_policy import ACTPolicy
from utils import TacArenaDataset,get_norm_stats
from encoder.clean_v2 import PREPROCESS,manifest_hash
from encoder.detail_v2 import load_encoder_checkpoint

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,value):
 path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def save(path,value):
 path=Path(path);tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)
def resolve_policy_config(config_path,tactile_ckpt,out):
 cfg=yaml.safe_load(Path(config_path).read_text()) or {}
 cfg.setdefault('seed',42)
 cfg.update(tactile_ckpt=str(tactile_ckpt),ckpt_dir=str(out),num_epochs=6000,device='cuda:0')
 return cfg
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--data',required=True);p.add_argument('--encoder-run',required=True);p.add_argument('--out',required=True);p.add_argument('--source-commit',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-batch',type=int,default=2);p.add_argument('--eval-smoke');p.add_argument('--resume',action='store_true');a=p.parse_args()
 if not a.smoke and (ROOT/'SOURCE_COMMIT').read_text().strip()!=a.source_commit:raise RuntimeError('Immutable release/source commit mismatch')
 out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True);lock=open(out/'run.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (out/'completion.json').exists():raise RuntimeError('Already completed')
 if (out/'training_state.pt').exists() and not a.resume:raise RuntimeError('Existing state: use --resume')
 er=Path(a.encoder_run).resolve();completion=json.loads((er/'completion.json').read_text())
 assert completion['status']=='completed'
 if not a.smoke:
  assert not completion.get('smoke',False) and not completion.get('probe',False)
  ev=json.loads(Path(a.eval_smoke).read_text());assert ev['status']=='smoke_pass','Need a validated simulator smoke path'
 ckpt=er/completion['checkpoint'];assert sha(ckpt)==completion['checkpoint_sha256']
 enc,_=load_encoder_checkpoint(ckpt,expected_encoder_type='detail_v2',expected_preprocess=PREPROCESS)
 cfg=resolve_policy_config(a.config,ckpt,out)
 if not a.smoke and cfg['num_steps']!=4000:raise ValueError('Formal protocol requires exactly 4000 optimizer updates')
 cfg.update(training_control='optimizer_updates',resume_data_order='restore independent sampler epoch generator state and batch cursor')
 cfg.update(run_id=out.name,source_commit=a.source_commit,encoder_sha256=sha(ckpt),smoke=a.smoke,physical_batch=a.smoke_batch if a.smoke else 32,gradient_accumulation=1 if a.smoke and a.smoke_batch==2 else 2)
 contract_hash=manifest_hash(cfg)
 if a.resume:
  previous=json.loads((out/'resolved_config.json').read_text());assert manifest_hash(previous)==contract_hash,'Resume contract mismatch'
 (out/'train_config.yml').write_text(yaml.safe_dump(cfg));atomic(out/'resolved_config.json',cfg)
 torch.set_num_threads(4);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
 episode_count=len(list(Path(a.data).glob('episode_*.hdf5')));assert episode_count==50
 indices=np.random.RandomState(1).permutation(episode_count);split={'seed':1,'train':indices[:40].tolist(),'val':indices[40:].tolist(),'normalization':'all50 per existing ACT protocol; validation not strict statistical holdout'};atomic(out/'policy_split.json',split)
 stats,_=get_norm_stats(a.data,episode_count)
 with (out/'dataset_stats.pkl').open('wb') as f:pickle.dump(stats,f)
 datasets={s:TacArenaDataset(np.array(split[s]),a.data,cfg['camera_names'],cfg['tactile_names'],stats,cfg['chunk_size']) for s in ('train','val')}
 batch=a.smoke_batch if a.smoke else 32
 train_generator=torch.Generator().manual_seed(cfg['seed'])
 loaders={s:DataLoader(ds,generator=train_generator if s=='train' else None,batch_size=batch,shuffle=s=='train',num_workers=0 if a.smoke else 4,pin_memory=True,drop_last=s=='train') for s,ds in datasets.items()}
 policy=ACTPolicy(cfg).cuda();optimizer=policy.configure_optimizers();tactile=policy.model.backbones[1].backbone
 managed={id(p) for g in optimizer.param_groups for p in g['params']}
 assert all(id(p) in managed for p in tactile.parameters() if p.requires_grad)
 group_report=[{'lr':g['lr'],'parameters':sum(p.numel() for p in g['params'])} for g in optimizer.param_groups]
 cam,tac,q,act,pad=next(iter(loaders['train']));cam,tac,q,act,pad=[v.cuda() for v in (cam,tac,q,act,pad)]
 policy.eval();enc=enc.cuda().eval()
 with torch.no_grad():
  torch.testing.assert_close(tactile(tac[:,0]),enc(tac[:,0]),rtol=0,atol=0)
  actions=policy(q,cam,tac);changed=policy(q,cam,torch.zeros_like(tac))
  delta=float((actions-changed).abs().max());assert delta>1e-8,'Tactile bypassed';assert actions.shape==(batch,50,8);assert torch.isfinite(actions).all()
 del enc
 policy.train();bn_before={n:b.clone() for n,b in tactile.named_buffers() if 'running_' in n}
 params_before={n:p.detach().clone() for n,p in tactile.named_parameters() if p.requires_grad}
 loss=policy(q,cam,tac,act,pad)['loss'];optimizer.zero_grad();loss.backward()
 grad={n:float(p.grad.norm()) for n,p in tactile.named_parameters() if p.requires_grad and p.grad is not None};assert grad and all(np.isfinite(list(grad.values())))
 optimizer.step();updates={n:float((p-params_before[n]).abs().max()) for n,p in tactile.named_parameters() if p.requires_grad}
 assert max(v for n,v in updates.items() if 'cross_attention' in n)>0
 for n,b in tactile.named_buffers():
  if n in bn_before:assert torch.equal(b,bn_before[n]),f'BN drift {n}'
 atomic(out/'interface_smoke.json',{'status':'passed','actions_shape':list(actions.shape),'tactile_perturbation_max_abs':delta,'encoder_sha256':sha(ckpt),'critical_weight_coverage':1.0,'optimizer_groups':group_report,'tactile_gradients':grad,'tactile_parameter_updates':updates,'bn_statistics':'frozen','bn_affine':'frozen'})
 # The interface step must never become an extra formal optimization step.
 torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed']);del policy,optimizer,tactile
 policy=ACTPolicy(cfg).cuda();optimizer=policy.configure_optimizers();step=0
 if a.resume:
  state=torch.load(out/'training_state.pt',weights_only=False);assert state['encoder_sha256']==sha(ckpt);assert state['contract_hash']==contract_hash;policy.load_state_dict(state['policy'],strict=True);optimizer.load_state_dict(state['optimizer']);step=state['step'];torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
 target=2 if a.smoke else cfg['num_steps'];accum=1 if a.smoke and a.smoke_batch==2 else 2
 train_generator.manual_seed(cfg['seed']);sampler_offset=0
 if a.resume:train_generator.set_state(state['sampler_epoch_state']);sampler_offset=state['sampler_offset']
 sampler_epoch_state=train_generator.get_state();iterator=iter(loaders['train'])
 for _ in range(sampler_offset):next(iterator)
 t0=time.time();micro=step*accum
 while step<target:
  policy.train();optimizer.zero_grad(set_to_none=True);train_loss=0.
  for _ in range(accum):
   try:batch_data=next(iterator)
   except StopIteration:
    sampler_epoch_state=train_generator.get_state();iterator=iter(loaders['train']);sampler_offset=0;batch_data=next(iterator)
   sampler_offset+=1
   cam,tac,q,act,pad=[v.cuda(non_blocking=True) for v in batch_data]
   parts=policy(q,cam,tac,act,pad);loss=parts['loss'];assert torch.isfinite(loss);(loss/accum).backward();train_loss+=float(loss)/accum;micro+=1
  optimizer.step();step+=1
  if step%25==0 or step==1:print(json.dumps({'step':step,'micro_iterations':micro,'loss':train_loss,'elapsed':time.time()-t0,'peak_memory':torch.cuda.max_memory_allocated(),'learning_rates':[g['lr'] for g in optimizer.param_groups],'physical_batch':batch,'effective_batch':batch*accum}),flush=True)
  if step%500==0 or step==target:
   policy.eval();vals=[]
   with torch.no_grad():
    for cam,tac,q,act,pad in loaders['val']:
     val=policy(q.cuda(),cam.cuda(),tac.cuda(),act.cuda(),pad.cuda())['loss'];vals.append(float(val))
     if a.smoke:break
   assert np.isfinite(vals).all()
   with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps({'step':step,'train_loss':train_loss,'val_loss':float(np.mean(vals))})+'\n')
   save(out/'policy_last.ckpt',policy.state_dict());save(out/'training_state.pt',{'policy':policy.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'encoder_sha256':sha(ckpt),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'contract_hash':contract_hash,'sampler_epoch_state':sampler_epoch_state,'sampler_offset':sampler_offset})
 assert step==target and micro==target*accum
 loaded=torch.load(out/'policy_last.ckpt',weights_only=True);policy.load_state_dict(loaded,strict=True)
 atomic(out/'completion.json',{'status':'completed','run_id':out.name,'source_commit':a.source_commit,'smoke':a.smoke,'optimizer_updates':step,'micro_iterations':micro,'encoder_sha256':sha(ckpt),'checkpoint':'policy_last.ckpt','checkpoint_sha256':sha(out/'policy_last.ckpt'),'dataset_stats_sha256':sha(out/'dataset_stats.pkl'),'policy_split_hash':manifest_hash(split),'elapsed':time.time()-t0})
 print('completed',out,flush=True)
if __name__=='__main__':main()

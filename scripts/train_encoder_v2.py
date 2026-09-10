"""Finite-budget P0 training; complete state and atomic encoder-only publication."""
import argparse,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from encoder.clean_v2 import CleanDataset,PREPROCESS,manifest_hash
from encoder.dynamic_data import DynamicCleanDataset
from encoder.detail_v2 import build_encoder,save_encoder_checkpoint,load_encoder_checkpoint,warm_start_dynamic_encoder
from encoder.pretrain_v2 import P0Model

def atomic_json(path,value):
 path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def validate_dynamic_audit(cfg,audit):
 if cfg['encoder_type']!='detail_v2_dynamic':return
 expected={'sequence_length':cfg['sequence_length'],'temporal_stride':cfg['temporal_stride']}
 if audit.get('history')!=expected:raise ValueError(f'dynamic audit history mismatch: expected {expected}, got {audit.get("history")}')
 for name in ('depth','marker','delta_marker'):
  statistics=audit.get('normalization',{}).get(name)
  if not isinstance(statistics,dict):raise ValueError(f'dynamic audit missing {name} normalization')
  if statistics.get('source')!='train_only':raise ValueError(f'dynamic audit {name} normalization must be train-only')
  if statistics.get('statistics_frame_stride')!=10:raise ValueError(f'dynamic audit {name} statistics stride must be 10')
  if not isinstance(statistics.get('count'),int) or statistics['count']<=0:raise ValueError(f'dynamic audit {name} count must be positive')
  if not np.isfinite(statistics.get('mean')) or not np.isfinite(statistics.get('std')) or statistics['std']<=0:raise ValueError(f'dynamic audit {name} statistics must be finite with positive std')
def initialize_encoder(cfg,*,spatial_init,spatial_source_commit,trunk_init=None):
 encoder_type=cfg['encoder_type']
 if encoder_type=='detail_v2_dynamic':
  if spatial_init is None:raise ValueError('detail_v2_dynamic requires --spatial-init')
  if not spatial_source_commit:raise ValueError('detail_v2_dynamic requires --spatial-source-commit')
  if trunk_init is not None:raise ValueError('detail_v2_dynamic uses --spatial-init, not --trunk-init')
  source=Path(spatial_init).resolve()
  if not source.is_file():raise FileNotFoundError(source)
  source_sha256=sha(source)
  load_encoder_checkpoint(source,expected_encoder_type='detail_v2',expected_preprocess=PREPROCESS)
  structure={k:cfg[k] for k in ('sequence_length','temporal_stride','token_dim','num_heads','attention_dropout','initial_gate') if k in cfg}
  encoder,report=warm_start_dynamic_encoder(source,**structure)
  if sha(source)!=source_sha256:raise RuntimeError('Spatial V2 source changed during warm-start')
  return encoder,{'spatial_init':str(source),'spatial_init_sha256':source_sha256,'spatial_source_commit':spatial_source_commit,'warm_start_report':report}
 trunk_state=torch.load(trunk_init,weights_only=True) if trunk_init else None
 return build_encoder(encoder_type,trunk_state=trunk_state),{}
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--clean',required=True);p.add_argument('--data-audit',required=True);p.add_argument('--split',required=True);p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--probe',action='store_true');p.add_argument('--source-commit');p.add_argument('--trunk-init');p.add_argument('--spatial-init');p.add_argument('--spatial-source-commit');p.add_argument('--resume',action='store_true');a=p.parse_args()
 cfg=json.loads(Path(a.config).read_text());cfg['source_commit']=a.source_commit or cfg['source_commit'];out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 import fcntl
 lock=open(out/'run.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (out/'completion.json').exists():raise RuntimeError('Already completed; refusing duplicate')
 if (out/'last_state.pt').exists() and not a.resume:raise RuntimeError('Use --resume for existing run')
 random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.set_num_threads(4)
 audit=json.loads(Path(a.data_audit).read_text());split=json.loads(Path(a.split).read_text());assert audit['split_hash']==manifest_hash(split)
 validate_dynamic_audit(cfg,audit)
 cfg.update(split_hash=audit['split_hash'],normalization=audit['normalization'],preprocess=PREPROCESS,run_id=out.name,pid=os.getpid(),smoke=a.smoke,probe=a.probe,data_audit_sha256=sha(a.data_audit))
 assert cfg['weights']=={k:1.0 for k in cfg['active_heads']},'P0 fixes all active weights to 1'
 assert set(cfg['active_heads']) <= {'depth','marker'},'This audited P0 supports contact targets only'
 encoder,initialization=initialize_encoder(cfg,spatial_init=a.spatial_init,spatial_source_commit=a.spatial_source_commit,trunk_init=a.trunk_init)
 cfg.update(initialization)
 contract={k:v for k,v in cfg.items() if k not in ('pid','source_commit')}
 contract_hash=manifest_hash(contract)
 if a.resume:
  previous=json.loads((out/'resolved_config.json').read_text());assert previous.get('contract_hash')==contract_hash,'Resume contract changed'
 cfg['contract_hash']=contract_hash
 atomic_json(out/'resolved_config.json',cfg)
 model=P0Model(encoder,cfg['active_heads'],dynamic_weight=cfg.get('dynamic_weight') if cfg['encoder_type']=='detail_v2_dynamic' else None).cuda()
 trunk=encoder.export_trunk_state();init=out/'shared_trunk_init.pt';temp=init.with_suffix('.tmp');torch.save(trunk,temp);temp.replace(init);cfg['shared_trunk_sha256']=sha(init)
 optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
 if cfg['encoder_type']=='detail_v2_dynamic':
  datasets={s:DynamicCleanDataset(a.clean,split[s],audit['normalization'],stride=cfg['frame_stride'],sequence_length=cfg['sequence_length'],temporal_stride=cfg['temporal_stride']) for s in ('train','val')}
 else:
  datasets={s:CleanDataset(a.clean,split[s],cfg['active_heads'],audit['normalization'],stride=cfg['frame_stride']) for s in ('train','val')}
 if a.smoke:
  ids=np.linspace(0,len(datasets['train'])-1,8,dtype=int).tolist();datasets['train']=Subset(datasets['train'],ids)
 loaders={s:DataLoader(ds,batch_size=8 if a.smoke else cfg['batch_size'],shuffle=s=='train',num_workers=0 if a.smoke else cfg['workers'],pin_memory=True,drop_last=False) for s,ds in datasets.items()}
 cfg['samples']={k:len(v) for k,v in datasets.items()};atomic_json(out/'resolved_config.json',cfg)
 best=float('inf');step=0;start=0
 if a.resume:
  state=torch.load(out/'last_state.pt',weights_only=False);assert state['split_hash']==cfg['split_hash'];assert state['contract_hash']==contract_hash;model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);start=state['epoch']+1;step=state['step'];best=state['best'];torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
 print(json.dumps({'status':'running','pid':os.getpid(),'config':cfg}),flush=True)
 t0=time.time();gradient_report={};curve=[]
 for epoch in range(start,1 if a.smoke or a.probe else cfg['epochs']):
  model.train();values=[]
  iterator=iter(loaders['train']);fixed=next(iterator) if a.smoke else None
  for bi in range(50 if a.smoke else (3 if a.probe else len(loaders['train']))):
   x,targets=fixed if a.smoke else next(iterator);x=x.cuda(non_blocking=True);targets={k:v.cuda(non_blocking=True) for k,v in targets.items()}
   optimizer.zero_grad(set_to_none=True);loss,parts=model.loss(x,targets)
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
   before={n:p.detach().clone() for n,p in encoder.named_parameters()} if step<3 else {}
   loss.backward()
   if step<3:
    gradient_report[str(step)]={n:float(p.grad.norm()) for n,p in encoder.named_parameters() if p.grad is not None}
    if any(not np.isfinite(v) for v in gradient_report[str(step)].values()):raise RuntimeError('Nonfinite gradient')
   optimizer.step();step+=1
   if before:
    updates={n:float((p.detach()-before[n]).abs().max()) for n,p in encoder.named_parameters()}
    atomic_json(out/f'gradient_step_{step}.json',{'gradients':gradient_report[str(step-1)],'parameter_max_update':updates})
   v=float(loss);values.append(v);curve.append(v)
   if bi%10==0:print(json.dumps({'epoch':epoch,'batch':bi,'step':step,'loss':v,'heads':{k:float(v) for k,v in parts.items()},'elapsed':time.time()-t0,'peak_memory':torch.cuda.max_memory_allocated()}),flush=True)
  model.eval();total=0.;count=0
  with torch.no_grad():
   for x,targets in loaders['val']:
    targets={k:v.cuda() for k,v in targets.items()};loss,_=model.loss(x.cuda(),targets);total+=float(loss)*len(x);count+=len(x)
    if a.smoke or a.probe:break
  val=total/count
  if not np.isfinite(val):raise RuntimeError('Nonfinite validation')
  if val<best:
   best=val;save_encoder_checkpoint(out/'best.pt',encoder,PREPROCESS)
  save_encoder_checkpoint(out/'last.pt',encoder,PREPROCESS)
  state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'step':step,'best':best,'split_hash':cfg['split_hash'],'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'contract_hash':contract_hash}
  temp=out/'last_state.tmp';torch.save(state,temp);temp.replace(out/'last_state.pt')
  record={'epoch':epoch,'step':step,'train_loss':sum(values)/len(values),'val_loss':val,'best_val':best,'elapsed':time.time()-t0}
  with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  print(json.dumps(record),flush=True)
 if a.smoke and np.mean(curve[-5:])>=np.mean(curve[:5]):raise AssertionError('Fixed real batch did not overfit')
 reloaded,meta=load_encoder_checkpoint(out/'best.pt',expected_encoder_type=cfg['encoder_type'],expected_preprocess=PREPROCESS)
 # Validate exported last matches the actual eval embedding (best may be from an earlier epoch).
 last,_=load_encoder_checkpoint(out/'last.pt',expected_encoder_type=cfg['encoder_type'],expected_preprocess=PREPROCESS);last=last.cuda().eval()
 check_x=next(iter(loaders['val']))[0].cuda()
 with torch.no_grad():torch.testing.assert_close(last(check_x),encoder.eval()(check_x),rtol=0,atol=0)
 atomic_json(out/'completion.json',{'status':'completed','run_id':out.name,'source_commit':cfg['source_commit'],'encoder_type':cfg['encoder_type'],'steps':step,'best_val':best,'checkpoint':'best.pt','checkpoint_sha256':sha(out/'best.pt'),'split_hash':cfg['split_hash'],'elapsed':time.time()-t0,'smoke':a.smoke,'probe':a.probe,'overfit_first5':float(np.mean(curve[:5])) if curve else None,'overfit_last5':float(np.mean(curve[-5:])) if curve else None,'contract_hash':contract_hash,**initialization})
if __name__=='__main__':main()

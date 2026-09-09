"""Persistent, fail-closed encoder → policy → eval orchestration for one run."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path

if not __debug__:raise RuntimeError('Validation requires Python without -O')

def read(p):return json.loads(Path(p).read_text())
def atomic(p,v):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check_encoder(root):
 d=read(root/'completion.json');cfg=read(root/'resolved_config.json')
 assert d['status']=='completed' and not d.get('smoke') and not d.get('probe')
 assert d['run_id']==root.name and d['encoder_type']=='detail_v2'
 assert d['steps']==cfg['epochs']*((cfg['samples']['train']+cfg['batch_size']-1)//cfg['batch_size'])
 assert sha(root/d['checkpoint'])==d['checkpoint_sha256']
 return d

def main():
 p=argparse.ArgumentParser();p.add_argument('--encoder-run',required=True);p.add_argument('--encoder-pid',type=int,required=True);p.add_argument('--policy-data',required=True);p.add_argument('--eval-smoke',required=True);p.add_argument('--external-runtime',required=True);p.add_argument('--out',required=True);p.add_argument('--gpu',required=True);p.add_argument('--source-commit',required=True);a=p.parse_args()
 source=Path(__file__).resolve().parents[1]
 if (source/'SOURCE_COMMIT').read_text().strip()!=a.source_commit:raise RuntimeError('Immutable release/source commit mismatch')
 if not a.gpu.isdigit():raise ValueError('One numeric physical GPU required')
 out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
 lock=open(out/'chain.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 er=Path(a.encoder_run).resolve();pr=out/'policy_seed42';state={'run_id':out.name,'pid':os.getpid(),'source_commit':a.source_commit,'encoder_run':str(er),'policy_run':str(pr),'stage':'encoder','status':'waiting'}
 def status(**kw):state.update(kw);atomic(out/'status.json',state);print(json.dumps(state),flush=True)
 def run(stage,argv,env):
  gpu_dir=Path.home()/'.cache/details_v2_gpu_locks';gpu_dir.mkdir(parents=True,exist_ok=True)
  gpu_lock=open(gpu_dir/f'gpu_{a.gpu}.lock','a');fcntl.flock(gpu_lock,fcntl.LOCK_EX)
  while True:
   row=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
   uuid=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
   processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
   if int(row)<256 and uuid not in processes:break
   status(stage=stage,status='pending_resource',gpu=a.gpu);time.sleep(10)
  log=out/f'{stage}.log'
  with log.open('a') as f:
   child=subprocess.Popen(argv,env=env,stdout=f,stderr=subprocess.STDOUT)
   status(stage=stage,status='running',child_pid=child.pid,log=str(log));rc=child.wait()
  if rc:raise RuntimeError(f'{stage} failed rc={rc}; see {log}')
 try:
  status()
  while not (er/'completion.json').exists():
   proc=Path(f'/proc/{a.encoder_pid}/cmdline')
   if not proc.exists() or b'train_encoder_v2' not in proc.read_bytes():raise RuntimeError('Encoder exited without verified completion')
   time.sleep(10)
  enc=check_encoder(er);status(encoder_sha256=enc['checkpoint_sha256'])
  assert read(a.eval_smoke)['status']=='smoke_pass'
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(source),OMP_NUM_THREADS='4')
  if not (pr/'completion.json').exists():
   argv=[sys.executable,str(source/'scripts/train_policy_v2.py'),'--config',str(source/'configs/policy_v2.yml'),'--data',a.policy_data,'--encoder-run',str(er),'--out',str(pr),'--source-commit',a.source_commit,'--eval-smoke',a.eval_smoke]
   if (pr/'training_state.pt').exists():argv.append('--resume')
   run('policy',argv,env)
  policy=read(pr/'completion.json');assert policy['status']=='completed' and not policy['smoke'];assert policy['source_commit']==a.source_commit;assert policy['encoder_sha256']==enc['checkpoint_sha256'];assert policy['optimizer_updates']==4000 and policy['micro_iterations']==8000;assert sha(pr/policy['checkpoint'])==policy['checkpoint_sha256']
  status(policy_sha256=policy['checkpoint_sha256'])
  # Separate outputs prevent double counting; full eval includes quick seeds and
  # is reported independently, never summed with the quick result.
  from scripts.prepare_eval_v2 import _source_tree_sha256
  expected_source_hash=_source_tree_sha256(source)
  for label,seeds in [('quick',list(range(20))),('full',list(range(100)))]:
   manifest=out/f'{label}_seeds.json';atomic(manifest,seeds);dest=out/f'eval_{label}'
   if (dest/'results/completion.json').exists():
    result=read(dest/'results/completion.json')
    if result.get('status')=='completed' and result.get('policy_checkpoint_sha256')==policy['checkpoint_sha256'] and result.get('requested_seeds')==seeds and result.get('source_tree_sha256')==expected_source_hash and result.get('train_config_sha256')==sha(pr/'train_config.yml'):continue
    raise RuntimeError(f'Existing eval output must be audited before reuse: {dest}')
   run('eval_'+label,[sys.executable,str(source/'scripts/prepare_eval_v2.py'),'--external-runtime',a.external_runtime,'--source',str(source),'--policy-run',str(pr),'--out',str(dest),'--gpu',str(a.gpu),'--seeds',str(manifest),'--launch'],env)
   result=read(dest/'results/completion.json');assert result['status']=='completed' and result['requested_seeds']==seeds and result['valid_episode_count']==len(seeds) and result['policy_checkpoint_sha256']==policy['checkpoint_sha256'] and result['source_tree_sha256']==expected_source_hash,f'Incomplete or mismatched {label} evaluation'
  status(stage='finished',status='completed')
 except Exception as e:
  status(status='failed',error=str(e));raise
if __name__=='__main__':main()

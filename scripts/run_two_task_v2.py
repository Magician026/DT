"""Serial task runner; reuses live encoder and never overwrites an eval attempt."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
from scripts.run_v2_chain import check_encoder
from scripts.summarize_v2_eval import write_outputs

def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--source',required=True);p.add_argument('--encoder-pid',type=int,required=True);a=p.parse_args()
 base=Path(a.base);source=Path(a.source);home=Path.home();commit=(source/'SOURCE_COMMIT').read_text().strip();gpu='2'
 lock=open(base/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state={'pid':os.getpid(),'source_commit':commit,'gpu':gpu}
 def status(**kw):
  state.update(kw,timestamp=datetime.now(timezone.utc).isoformat());tmp=base/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(base/'status.json');print(json.dumps(state),flush=True)
 def run(stage,argv):
  ld=home/'.cache/details_v2_gpu_locks';ld.mkdir(parents=True,exist_ok=True)
  with (ld/'gpu_2.lock').open('a') as gl:
   fcntl.flock(gl,fcntl.LOCK_EX)
   while True:
    uuid=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    mem=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    if uuid not in procs and mem<256:break
    status(stage=stage,status='waiting_gpu');time.sleep(30)
   log=base/'runs'/state['task']/(stage+'.log');env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONPATH=str(source),OMP_NUM_THREADS='4',CONDA_SH=str(home/'miniconda3/etc/profile.d/conda.sh'),CONDA_ENV='UniVTAC',ISAAC_SIM_ROOT=str(home/'isaacsim-4.5.0'))
   with log.open('a') as f:
    child=subprocess.Popen(argv,env=env,stdout=f,stderr=subprocess.STDOUT);status(stage=stage,status='running',child_pid=child.pid,log=str(log));rc=child.wait()
   if rc:raise RuntimeError(f'{stage} exited {rc}; retained log {log}')
 def cmd(script,*args):return [sys.executable,str(source/'scripts'/script),*map(str,args)]
 try:
  for task in ('insert_tube','pull_out_key'):
   status(task=task,stage='audit',status='checking');out=base/'runs'/task;out.mkdir(parents=True,exist_ok=True);audit=base/'audits'/('pull_out_key_depth_only' if task=='pull_out_key' else task)
   if not (audit/'data_audit.json').exists():raise RuntimeError(f'{task} data audit not complete: marker semantics diagnosis required; do not train')
   enc=out/'encoder';cfg=source/'configs'/f'encoder_{task}_v2.json';policy_cfg=source/'configs'/f'policy_{task}_v2_seed0.yml'
   ec=cmd('train_encoder_v2.py','--config',cfg,'--clean',home/f'UniVTAC details/data/{task}/clean','--data-audit',audit/'data_audit.json','--split',audit/'split.json','--source-commit',commit)
   if task=='insert_tube' and not (enc/'completion.json').exists():
    status(stage='encoder',status='waiting_existing',child_pid=a.encoder_pid,log=str(out/'encoder.log'))
    while not (enc/'completion.json').exists():
     proc=Path(f'/proc/{a.encoder_pid}/cmdline')
     if not proc.exists() or b'train_encoder_v2' not in proc.read_bytes():raise RuntimeError('Existing encoder exited without completion')
     time.sleep(30)
   if not (enc/'completion.json').exists():
    es=out/'encoder_smoke'
    if not (es/'completion.json').exists():run('encoder_smoke',ec+['--out',str(es),'--smoke'])
    assert read(es/'completion.json')['smoke']
    run('encoder',ec+['--out',str(enc)])
   e=check_encoder(enc);assert read(enc/'resolved_config.json')['seed']==read(cfg)['seed']
   pc=cmd('train_policy_v2.py','--config',policy_cfg,'--data',home/f'UniVTAC details/policy/ACT/data/sim-{task}/clean-50','--encoder-run',enc,'--source-commit',commit)
   sm=out/'policy_smoke'
   if not (sm/'completion.json').exists():run('policy_smoke',pc+['--out',str(sm),'--smoke','--smoke-batch','32'])
   seeds=out/'eval_seeds.json';seeds.write_text(json.dumps(list(range(1000000,1000100))))
   def ev(pr,dest,smoke=False):
    completion=dest/'results/completion.json'
    if completion.exists():
     c=read(completion);assert c['status']==('smoke_pass' if smoke else 'completed');assert c['policy_checkpoint_sha256']==read(pr/'completion.json')['checkpoint_sha256'];return
    if dest.exists():raise RuntimeError('Existing incomplete evaluation: audit before reuse')
    argv=cmd('prepare_eval_v2.py','--external-runtime',home/'UniVTAC details/reproduction_official_20260905/eval_runtime_official','--source',source,'--policy-run',pr,'--out',dest,'--gpu',gpu,'--seeds',seeds,'--launch')
    if smoke:argv+=['--smoke-steps','30','--allow-smoke']
    run(dest.name,argv)
   smoke=out/'eval_smoke_30';ev(sm,smoke,True);pr=out/'policy_seed0'
   if not (pr/'completion.json').exists():run('policy_seed0',pc+['--out',str(pr),'--eval-smoke',str(smoke/'results/completion.json')])
   c=read(pr/'completion.json');assert c['optimizer_updates']==4000 and c['micro_iterations']==8000 and c['encoder_sha256']==e['checkpoint_sha256'];assert read(pr/'resolved_config.json')['seed']==0
   dest=out/'eval_1000000_1000099';ev(pr,dest)
   s=write_outputs([dest],range(1000000,1000100),base/'results',f'{task}_v2_policy_seed0');status(stage='task_completed',status='completed',success=s['success_count'])
  status(stage='all_completed',status='completed')
 except Exception as e:status(status='failed',error=str(e));raise
if __name__=='__main__':main()

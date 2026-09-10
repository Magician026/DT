"""Finish the existing Key policy without restarting it; publish eval under new root."""
import os,json,time,hashlib,shutil,subprocess,sys
from pathlib import Path
home=Path.home();old=home/'details_v2_tasks_20260910';base=home/'UniVTAC details v2/details_v2_tasks_20260910';source=base/'releases/smoke30';pr=old/'runs/pull_out_key/policy_seed0';dest=base/'runs/pull_out_key/policy_seed0'
def status(stage,**kw):
 p=base/'migration_status.json';p.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),timestamp=time.time(),**kw),indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
try:
 status('waiting_existing_policy',training_pid=3978)
 while not (pr/'completion.json').exists():
  proc=Path('/proc/3978/cmdline')
  if not proc.exists() or b'train_policy_v2' not in proc.read_bytes():raise RuntimeError('Policy exited without completion')
  time.sleep(30)
 c=json.loads((pr/'completion.json').read_text());assert c['status']=='completed' and c['optimizer_updates']==4000 and c['micro_iterations']==8000
 assert sha(pr/'policy_last.ckpt')==c['checkpoint_sha256']
 while Path('/proc/3978/cmdline').exists() and Path('/proc/3978/cmdline').read_bytes():time.sleep(10)
 dest.mkdir(parents=True,exist_ok=False)
 for name in ('policy_last.ckpt','completion.json','dataset_stats.pkl','train_config.yml','resolved_config.json'):shutil.copy2(pr/name,dest/name)
 er=base/'runs/pull_out_key/encoder';er.mkdir(parents=True,exist_ok=True)
 for name in ('best.pt','completion.json','resolved_config.json'):shutil.copy2(old/'runs/pull_out_key/encoder'/name,er/name)
 assert sha(er/'best.pt')==c['encoder_sha256']
 # Preserve exact training config; its existing encoder path remains valid until eval ends.
 seeds=base/'key_eval_seeds.json';seeds.write_text(json.dumps(list(range(1000000,1000100))))
 env=os.environ.copy();env.update(PYTHONPATH=str(source),CONDA_SH=str(home/'miniconda3/etc/profile.d/conda.sh'),CONDA_ENV='UniVTAC',ISAAC_SIM_ROOT=str(home/'isaacsim-4.5.0'))
 ev=base/'runs/pull_out_key/eval_1000000_1000099'
 with (base/'key_eval_launcher.log').open('a') as f:
  child=subprocess.Popen([sys.executable,str(source/'scripts/prepare_eval_v2.py'),'--external-runtime',str(home/'UniVTAC details/reproduction_official_20260905/eval_runtime_official'),'--source',str(source),'--policy-run',str(dest),'--out',str(ev),'--gpu','2','--seeds',str(seeds),'--launch'],env=env,stdout=f,stderr=subprocess.STDOUT)
  status('eval_running',child_pid=child.pid);rc=child.wait()
  if rc:raise RuntimeError(f'Eval exit {rc}')
 sys.path.insert(0,str(source));from scripts.summarize_v2_eval import write_outputs
 write_outputs([ev],range(1000000,1000100),base/'results','pull_out_key_v2_policy_seed0');status('completed')
except Exception as e:status('failed',error=str(e));raise

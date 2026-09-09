"""One-task overnight sequence. Never restarts a live eval or changes its inputs."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
from summarize_v2_eval import write_outputs

def read(p):return json.loads(Path(p).read_text())
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def alive(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False

def morning(base,state):
 results=base/'results';first=results/'detail_v2_eval_0_99.json';second=results/'detail_v2_policy_seed_1_eval_0_99.json';quick=results/'detail_v2_eval_0_19.json'
 d=read(first) if first.exists() else (read(quick) if quick.exists() else None)
 b=read(second) if second.exists() else None
 count=d['total_count'] if d else 0
 status=f"{count} / 100 verified seeds completed"
 if d:
  ci=d['wilson95'];value=f"Success: {d['success_count']} / {count}\n\nSuccess Rate: {d['success_rate']:.1%}\n\n95% Wilson CI: [{ci[0]:.1%}, {ci[1]:.1%}]"
 else:value='Not yet available.'
 paired='MATCHED_BASELINE_NOT_AVAILABLE. Same-P0 original encoder exists, but no matched trained B0 policy was found; historical policies are reference only.'
 seed1=state.get('replica_training','not started')
 text=f'''# 1 Current Status

Updated: {datetime.now(timezone.utc).isoformat()}

Encoder: completed (V2 and same-data B0). Policy seed 42: completed, 4000 updates.\n\nEval: {status}. Night stage: {state.get('stage')}; status: {state.get('status')}.

# 2 Details V2 Result

{value}

Task: Insert HDMI. Official environment action limit: 600; unchanged for all nightly runs. Current source policy commit: 3cd75efd5b80aa14bc0ba13900c9b2e6e942bea8.

# 3 Matched B0 Result

{paired}

# 4 Policy Training Seeds

- Seed 42: training completed; checkpoint `v2_chain_seed42/policy_seed42/policy_last.ckpt`; {value.splitlines()[0]}
- Seed 1: training {seed1}; checkpoint expected `night_control/policy_seed1/policy_last.ckpt`; 100-seed result {f"{b['success_count']}/100 ({b['success_rate']:.1%})" if b else 'not yet available'}.
'''
 if b and first.exists():text+=f"\nMean success rate across the two policy seeds: {(d['success_rate']+b['success_rate'])/2:.1%}. Separate policy rates retained; not a pooled independent-training estimate.\n"
 text+='''
# 5 Integrity Audit

- V2 attention parameters trained? YES — Q/K/V gradients and parameter updates verified.
- V2 checkpoint loaded correctly? YES — strict state load, critical coverage 100%, checkpoint hashes recorded.
- Policy uses V2 full embedding? YES — canonical encoder and tactile perturbation changes ACT actions.
- Output [B,512]? YES.
- Random-init fallback? NO — missing checkpoint/statistics rejected.
- Encoder frozen or fine-tuned? Fine-tuned trunk and attention; BN affine/running statistics frozen to match reference policy.

# 6 Jobs / Processes

'''+f"Current stage: `{state.get('stage')}`; controller PID `{state.get('controller_pid')}`; child PID `{state.get('child_pid')}`; GPU `{state.get('gpu',2)}`. Log: `{state.get('log_name','night_control.log')}`. These are timestamped observations, not proof of perpetual liveness.\n"
 text+='''
# 7 Failures

- Initial eval preparation: PermissionError writing a copied read-only release. Fixed in 35a8a12 by making only the isolated copy writable; external symlinks unchanged. No scientific configuration changed.
- Earlier recovery wrapper: unmatched parenthesis before simulator launch; corrected and logs retained.
- Prior launcher started a duplicate full 0–99 attempt after successful quick 0–19, before the latest no-repeat instruction. Duplicate attempt stopped; all its rows excluded. Quick results preserved; continuation runs only 20–99. SIGTERM did not terminate that simulator; SIGKILL was required for that duplicate process only. Continuation startup briefly overlapped process teardown before its first rollout; no completed quick rollout was interrupted.
- Documentation previously said 300 actions; actual HDMI source specifies 600. Documentation corrected; environment and criterion unchanged.
'''
 if state.get('error'):text+=f"\nCurrent error: {state['error']}\n"
 text+=f'''
# 8 Git

Night controller source commit: `{state.get('source_commit','pending')}`. Public stage results and latest successful push are recorded in `results/git_sync.json`; if absent, final sync is pending. Do not infer push success from a local commit.

# 9 Evidence-backed Conclusion

INCONCLUSIVE — current results can establish pipeline operation and estimate this policy's success rate. A matched B0 policy comparison and independent policy seed are needed before claiming the architecture improves success.

# 10 Recommended Next Action

1. Complete and verify disjoint seed 0–99 results for each authorized policy seed.
2. Obtain a strictly matched B0 policy before drawing an architecture improvement conclusion.
3. Review paired results and policy-seed variability; keep negative outcomes without tuning this experiment.
'''
 (base/'MORNING_REPORT_DETAIL_V2.md').write_text(text)

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--initial-eval-pid',required=True,type=int);p.add_argument('--external-runtime',required=True);p.add_argument('--policy-data',required=True);p.add_argument('--source',required=True);p.add_argument('--gpu',default='2');p.add_argument('--poll',type=int,default=30);a=p.parse_args()
 base=Path(a.base).resolve();source=Path(a.source).resolve();out=base/'runs/night_control';out.mkdir(parents=True,exist_ok=True)
 owner=open(out/'owner.lock','a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
 commit=(source/'SOURCE_COMMIT').read_text().strip();state={'controller_pid':os.getpid(),'source_commit':commit,'gpu':a.gpu,'replica_training':'not started'}
 results=base/'results';chain=base/'runs/v2_chain_seed42';pr=out/'policy_seed1'
 def status(stage,kind,**kw):
  state.update(stage=stage,status=kind,timestamp=datetime.now(timezone.utc).isoformat(),**kw);atomic(out/'status.json',state);morning(base,state);print(json.dumps(state),flush=True)
 def wait_existing(directory,pid):
  while not (directory/'results/completion.json').exists():
   if not alive(pid):raise RuntimeError(f'{directory.name} exited without complete evidence')
   time.sleep(a.poll)
  c=read(directory/'results/completion.json')
  if c['status']!='completed':raise RuntimeError(f'{directory.name}: incomplete or infrastructure errors; retain evidence and repair before proceeding')
  while alive(pid):time.sleep(a.poll)
 def run(stage,argv):
  lockdir=Path.home()/'.cache/details_v2_gpu_locks';lockdir.mkdir(parents=True,exist_ok=True)
  with (lockdir/f'gpu_{a.gpu}.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   while True:
    uuid=subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    mem=int(subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    if uuid not in procs and mem<256:break
    status(stage,'pending_resource');time.sleep(a.poll)
   log=out/f'{stage}.log';env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu,PYTHONPATH=str(source),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
   with log.open('a') as f:
    child=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,env=env);status(stage,'running',child_pid=child.pid,log_name=f'night_control/{stage}.log');rc=child.wait()
   if rc:raise RuntimeError(f'{stage} failed return code {rc}; inspect retained log')
 def eval_stage(label,seeds):
  manifest=out/f'{label}_seeds.json';atomic(manifest,seeds);dest=out/label
  if (dest/'results/completion.json').exists():
   c=read(dest/'results/completion.json');policy=read(pr/'completion.json')
   if c['status']=='completed' and c['requested_seeds']==seeds and c['policy_checkpoint_sha256']==policy['checkpoint_sha256']:return dest
   raise RuntimeError('Existing eval attempt requires audit; refusing overwrite')
  run(label,[sys.executable,str(source/'scripts/prepare_eval_v2.py'),'--external-runtime',a.external_runtime,'--source',str(source),'--policy-run',str(pr),'--out',str(dest),'--gpu',a.gpu,'--seeds',str(manifest),'--launch']);return dest
 try:
  status('v2_seed42_eval_20_99','waiting_existing',child_pid=a.initial_eval_pid,log_name='v2_chain_seed42/eval_20_99/eval.log')
  wait_existing(chain/'eval_20_99',a.initial_eval_pid)
  first=write_outputs([chain/'eval_quick',chain/'eval_20_99'],range(100),results,'detail_v2_eval_0_99')
  (results/'detail_v2_eval_summary.md').write_text((results/'detail_v2_eval_0_99_summary.md').read_text())
  status('v2_seed42_aggregation','completed',first_success_count=first['success_count'])
  if c := read(chain/'eval_20_99/results/completion.json'):
   if c['encoder_sha256']!=first['encoder_sha256']:raise RuntimeError('Encoder mismatch')
  # Audit explicitly records no matched B0 policy. Never substitute historical results.
  audit=(results/'baseline_audit.md').read_text()
  if 'MATCHED_BASELINE_NOT_AVAILABLE' not in audit:raise RuntimeError('A matched baseline decision needs review before replica stage')
  if first['success_count']==0:raise RuntimeError('0/100 first-policy success: diagnose before spending on replica; do not tune automatically')
  if first['infrastructure_error_count']>0:raise RuntimeError('Infrastructure errors require audit before replica training')
  if not (pr/'completion.json').exists():
   state['replica_training']='running'
   argv=[sys.executable,str(source/'scripts/train_policy_v2.py'),'--config',str(source/'configs/policy_v2_seed1.yml'),'--data',a.policy_data,'--encoder-run',str(base/'runs/encoder_v2_seed42'),'--out',str(pr),'--source-commit',commit,'--eval-smoke',str(chain.parent/'eval_smoke_final_encoder/results/completion.json')]
   if (pr/'training_state.pt').exists():argv.append('--resume')
   run('policy_seed1',argv)
  c=read(pr/'completion.json');cfg=read(pr/'resolved_config.json')
  if c['optimizer_updates']!=4000 or c['micro_iterations']!=8000 or cfg['seed']!=1 or c['encoder_sha256']!=first['encoder_sha256']:raise RuntimeError('Replica training contract mismatch')
  state['replica_training']='completed';status('policy_seed1','completed')
  quick=eval_stage('seed1_eval_0_19',list(range(20)))
  write_outputs([quick],range(20),results,'detail_v2_policy_seed_1_eval_0_19')
  tail=eval_stage('seed1_eval_20_99',list(range(20,100)))
  second=write_outputs([quick,tail],range(100),results,'detail_v2_policy_seed_1_eval_0_99')
  atomic(results/'policy_seed_comparison.json',{'policy_seed_42':first['success_rate'],'policy_seed_1':second['success_rate'],'mean_success_rate':(first['success_rate']+second['success_rate'])/2,'encoder_seed':42,'note':'Two independent policy training seeds, one fixed encoder. Do not pool as independent model trainings.'})
  status('night_completed','completed')
 except Exception as e:
  status(state.get('stage','unknown'),'failed',error=str(e));raise
if __name__=='__main__':main()

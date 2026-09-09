"""Validate disjoint real rollout evidence; aggregate without duplicate counting."""
import argparse,csv,json,math
from datetime import datetime,timezone
from pathlib import Path

CONTRACT=('policy_checkpoint_sha256','encoder_sha256','train_config_sha256','source_tree_sha256','task')
def wilson(success,total):
 if not total:return None
 z=1.959963984540054;p=success/total;den=1+z*z/total
 center=(p+z*z/(2*total))/den;half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
 return [center-half,center+half]
def aggregate(run_dirs,expected):
 rows={};contexts=[]
 for directory in run_dirs:
  directory=Path(directory);result=directory/'results';manifest=json.loads((directory/'manifest.json').read_text())
  completion=json.loads((result/'completion.json').read_text())
  if completion['status']!='completed':raise ValueError(f'Incomplete evaluation {directory.name}')
  items=[json.loads(line) for line in (result/'outcomes.jsonl').read_text().splitlines() if line]
  requested=manifest['requested_seeds']
  if len(items)!=len(requested) or sorted(r['seed'] for r in items)!=sorted(requested):raise ValueError('Missing/duplicate/unrequested seed')
  if completion['requested_seeds']!=requested:raise ValueError('Completion seed mismatch')
  if contexts and any(completion[k]!=contexts[0][k] for k in CONTRACT):raise ValueError('Scientific contract drift')
  if contexts and manifest['official_env_hashes']!=contexts[0]['official_env_hashes']:raise ValueError('Environment drift')
  for row in items:
   seed=row['seed']
   if seed in rows:raise ValueError(f'Duplicate seed {seed}')
   if row['success'] not in (0,1) or row['termination'] not in ('success','timeout','early_stop'):raise ValueError(f'Invalid seed {seed}')
   if row['actions']<=0 or row['tactile_observations']!=row['actions']:raise ValueError('Not a verified real rollout')
   if bool(row['success'])!=(row['termination']=='success'):raise ValueError('Success/termination mismatch')
   rows[seed]={'seed':seed,'success':row['success'],'termination/status':row['termination'],'checkpoint':completion['policy_checkpoint_sha256'],'log_path':f'{directory.name}/results/log.log','actions':row['actions'],'tactile_changes':row['tactile_changes']}
  if sum(r['success'] for r in items)!=completion['success_count'] or len(items)!=completion['valid_episode_count']:raise ValueError('Counts mismatch')
  contexts.append({**completion,'official_env_hashes':manifest['official_env_hashes'],'evaluation_command':f'insert_HDMI demo ACT/deploy --total_num {len(requested)} --start_seed {requested[0]}','episode_action_limit':600})
 if set(rows)!=set(expected):raise ValueError('Unexpected aggregate seed set')
 successes=[s for s,r in sorted(rows.items()) if r['success']];failures=[s for s,r in sorted(rows.items()) if not r['success']]
 summary={k:contexts[0][k] for k in CONTRACT};summary.update(policy_source_commit=contexts[0]['policy_source_commit'],success_count=len(successes),total_count=len(rows),success_rate=len(successes)/len(rows),wilson95=wilson(len(successes),len(rows)),successful_seeds=successes,failed_seeds=failures,abnormal_seeds=[],infrastructure_error_count=sum(c['infrastructure_error_count'] for c in contexts),episode_action_limit=600,evaluation_commands=[c['evaluation_command'] for c in contexts],timestamp=datetime.now(timezone.utc).isoformat())
 return [rows[s] for s in sorted(rows)],summary

def write_outputs(run_dirs,expected,out,stem):
 rows,summary=aggregate(run_dirs,expected);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 with (out/f'{stem}.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 (out/f'{stem}.json').write_text(json.dumps(summary,indent=2)+'\n')
 ci=summary['wilson95'];text=f"# {stem}\n\nSuccess: {summary['success_count']} / {summary['total_count']}\n\nSuccess rate: {summary['success_rate']:.1%}; Wilson 95% CI: [{ci[0]:.1%}, {ci[1]:.1%}].\n\n"
 for key,value in summary.items():text+=f'- {key}: `{json.dumps(value)}`\n'
 text+='\nThis describes one trained policy. It does not establish an encoder architecture improvement.\n'
 (out/f'{stem}_summary.md').write_text(text);return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,required=True);p.add_argument('--out',required=True);p.add_argument('--stem',required=True);a=p.parse_args();print(json.dumps(write_outputs(a.runs,range(a.start,a.stop),a.out,a.stem),indent=2))

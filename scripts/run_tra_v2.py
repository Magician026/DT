"""Run the single authorized TRA policy, then fixed-seed quick/full triage."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def quick_decision(successes, valid, errors=0):
    if valid != 20 or errors or type(successes) is not int or not 0 <= successes <= valid:
        raise ValueError('Quick triage requires exactly 20 valid episodes and no infrastructure errors')
    return 'stop' if successes <= 5 else 'review_six' if successes == 6 else 'full100'


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def gpu_state(gpu):
    raw = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=uuid,memory.free,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    uuid, free, used, util = [s.strip() for s in raw.strip().split(',')]
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                        '--format=csv,noheader,nounits'], text=True)
    return {'uuid':uuid, 'free_mib':int(free), 'used_mib':int(used), 'util':int(util),
            'compute_processes':[line for line in processes.splitlines() if line.split(',')[0].strip()==uuid]}


def wait_gpu(gpu, *, evaluation):
    previous = None
    while True:
        state = gpu_state(gpu)
        available = (not state['compute_processes'] and state['used_mib'] < 200 and state['util'] <= 5) if evaluation else state['free_mib'] > 12288
        if available:
            print(json.dumps({'event':'gpu_available', 'evaluation':evaluation, **state}), flush=True)
            return state
        status = (evaluation, tuple(state['compute_processes']))
        if status != previous:
            print(json.dumps({'event':'waiting_for_gpu', 'evaluation':evaluation, **state}), flush=True)
            previous = status
        time.sleep(60)


def run(command, log, environment):
    print(json.dumps({'event':'launch', 'command':list(map(str,command)), 'log':str(log)}), flush=True)
    with log.open('ab', buffering=0) as stream:
        subprocess.run(list(map(str,command)), env=environment, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--external-runtime', required=True)
    parser.add_argument('--gpu', default='2')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    source = Path(__file__).resolve().parents[1]
    commit = (source/'SOURCE_COMMIT').read_text().strip()
    state_path = root/'results/pipeline_status.json'
    if state_path.exists():
        raise RuntimeError('TRA pipeline already launched; inspect artifacts instead of duplicating')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONPATH=str(source), OMP_NUM_THREADS='4')
    policy_run = root/'runs/policy_tra_v1_hdmi'
    def status(stage, **extra):
        record = {'stage':stage, 'source_commit':commit, 'time':time.time(), **extra}
        write_json(state_path, record)
        print(json.dumps(record), flush=True)
    def evaluate(name, seeds):
        status(name+'_waiting_for_idle_gpu')
        wait_gpu(args.gpu, evaluation=True)
        out = root/'runs'/name
        status(name, output=str(out))
        run([sys.executable, source/'scripts/prepare_eval_v2.py',
            '--external-runtime', args.external_runtime, '--source', source,
            '--policy-run', policy_run, '--out', out, '--gpu', args.gpu,
            '--seeds', seeds, '--launch'], root/'logs'/f'{name}.launch.log', env)
        completion = json.loads((out/'results/completion.json').read_text())
        outcomes = [json.loads(line) for line in (out/'results/outcomes.jsonl').read_text().splitlines()]
        expected = json.loads(Path(seeds).read_text())
        assert [row['seed'] for row in outcomes] == expected
        assert completion['valid_episode_count'] == len(expected)
        assert completion['success_count'] == sum(row['success'] for row in outcomes)
        assert completion['infrastructure_error_count'] == 0
        completion['outcomes_sha256'] = hashlib.sha256((out/'results/outcomes.jsonl').read_bytes()).hexdigest()
        write_json(root/'results'/f'{name}.json', completion)
        return completion
    try:
        status('training_waiting_for_gpu')
        wait_gpu(args.gpu, evaluation=False)
        status('training', output=str(policy_run))
        run([sys.executable, source/'scripts/train_policy_v2.py',
            '--config', source/'configs/policy_tra_v2_hdmi.yml', '--data', args.data,
            '--encoder-run', root/'runs/encoder_tra_init', '--out', policy_run,
            '--source-commit', commit, '--eval-smoke',
            root/'runs/eval_smoke_tra_v1/results/completion.json'], root/'logs/policy_tra_v1_hdmi.log', env)
        policy = json.loads((policy_run/'completion.json').read_text())
        assert policy['optimizer_updates'] == 4000 and policy['micro_iterations'] == 8000
        assert policy['dataset_stats_sha256'] == '9a92d7fe5873612d71240ecc667679ea517aadf0561e640a7d68571f02fc272b'
        assert policy['spatial_freeze_updates_applied'] == 400
        assert policy['alpha_initial']['raw'] != policy['alpha_final']['raw']
        quick = evaluate('eval_quick20_tra_v1_hdmi', root/'results/seeds_quick20.json')
        decision = quick_decision(quick['success_count'], quick['valid_episode_count'], quick['infrastructure_error_count'])
        write_json(root/'results/quick_decision.json', {'decision':decision, 'successes':quick['success_count'],
                   'valid':20, 'thresholds':'<=5 stop; 6 bounded review; >=7 full100'})
        if decision != 'full100':
            status('stopped_after_quick' if decision=='stop' else 'six_requires_curve_failure_review', quick_successes=quick['success_count'])
            return
        full = evaluate('eval_full100_tra_v1_hdmi', root/'results/seeds_full100.json')
        status('completed', quick_successes=quick['success_count'], full_successes=full['success_count'],
               conclusion='temporal_architecture_no_improvement' if full['success_count']<=31 else 'below_minimum' if full['success_count']==32 else 'minimum_accepted' if full['success_count']<35 else 'target_reached')
    except BaseException as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()

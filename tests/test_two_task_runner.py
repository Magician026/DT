import ast
from pathlib import Path
import yaml

def test_task_policy_contracts():
 root=Path(__file__).resolve().parents[1]
 for task,cameras in [('insert_tube',['cam_high','cam_wrist']),('pull_out_key',['cam_high'])]:
  cfg=yaml.safe_load((root/'configs'/f'policy_{task}_v2_seed0.yml').read_text())
  assert cfg['seed']==0 and cfg['camera_names']==cameras
  assert cfg['num_steps']==4000 and cfg['batch_size']==64
  assert cfg['task_name']==f'sim-{task}-clean-50'
  assert cfg['tactile_encoder_type']=='detail_v2'

def test_runner_has_no_hardcoded_hdmi_or_seed42_policy():
 text=(Path(__file__).resolve().parents[1]/'scripts/run_two_task_v2.py').read_text()
 ast.parse(text)
 assert 'insert_HDMI' not in text and 'policy_seed42' not in text
 assert 'range(1000000,1000100)' in text

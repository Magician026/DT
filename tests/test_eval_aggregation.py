import json
import pytest
from scripts.summarize_v2_eval import aggregate,wilson

def fixture_run(root,seeds):
 root.mkdir();r=root/'results';r.mkdir()
 context=dict(policy_checkpoint_sha256='p',encoder_sha256='e',train_config_sha256='c',source_tree_sha256='s',task='insert_HDMI',policy_source_commit='git',status='completed',requested_seeds=seeds,valid_episode_count=len(seeds),success_count=len(seeds),infrastructure_error_count=0)
 (r/'completion.json').write_text(json.dumps(context));(root/'manifest.json').write_text(json.dumps(dict(requested_seeds=seeds,official_env_hashes={'env':'h'})))
 (r/'outcomes.jsonl').write_text('\n'.join(json.dumps(dict(seed=s,success=1,termination='success',actions=3,tactile_observations=3,tactile_changes=2)) for s in seeds));return root

def test_disjoint_aggregation_and_wilson(tmp_path):
 a=fixture_run(tmp_path/'a',[0]);b=fixture_run(tmp_path/'b',[1]);rows,s=aggregate([a,b],range(2));assert len(rows)==2 and s['success_count']==2;assert 0<wilson(5,20)[0]<.25<wilson(5,20)[1]<1

def test_reject_duplicate_and_contract_drift(tmp_path):
 a=fixture_run(tmp_path/'a',[0]);b=fixture_run(tmp_path/'b',[0])
 with pytest.raises(ValueError,match='Duplicate'):aggregate([a,b],[0])
 d=json.loads((b/'results/completion.json').read_text());d['encoder_sha256']='wrong';(b/'results/completion.json').write_text(json.dumps(d))
 with pytest.raises(ValueError,match='contract'):aggregate([a,b],[0])

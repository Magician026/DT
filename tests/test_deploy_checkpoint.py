from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'policy/ACT'))
import act_policy

def test_deployment_rejects_missing_weights_before_model_creation(tmp_path):
 assert hasattr(act_policy,'require_deploy_artifacts'),'deployment must fail closed before creating model'
 for path in (None,tmp_path):
  with pytest.raises((ValueError,FileNotFoundError)):act_policy.require_deploy_artifacts(path)
 (tmp_path/'policy_last.ckpt').write_bytes(b'weights')
 with pytest.raises(FileNotFoundError):act_policy.require_deploy_artifacts(tmp_path)
 (tmp_path/'dataset_stats.pkl').write_bytes(b'stats')
 assert act_policy.require_deploy_artifacts(tmp_path)==tmp_path

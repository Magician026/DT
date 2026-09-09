import hashlib,json
import pytest
from scripts.run_v2_chain import check_encoder

@pytest.fixture
def run(tmp_path):
 p=tmp_path/'run';p.mkdir();(p/'best.pt').write_bytes(b'complete checkpoint payload')
 cfg={'epochs':5,'samples':{'train':8},'batch_size':4}
 (p/'resolved_config.json').write_text(json.dumps(cfg))
 d={'status':'completed','run_id':'run','encoder_type':'detail_v2','steps':10,'checkpoint':'best.pt','checkpoint_sha256':hashlib.sha256((p/'best.pt').read_bytes()).hexdigest(),'smoke':False,'probe':False}
 (p/'completion.json').write_text(json.dumps(d));return p,d

def test_completed_encoder_accepts_exact_budget_and_hash(run):
 p,d=run;assert check_encoder(p)==d

@pytest.mark.parametrize('field,value',[('smoke',True),('probe',True),('steps',3),('run_id','stale'),('encoder_type','original'),('checkpoint_sha256','wrong')])
def test_rejects_diagnostic_stale_or_partial_encoder(run,field,value):
 p,d=run;d[field]=value;(p/'completion.json').write_text(json.dumps(d))
 with pytest.raises(AssertionError):check_encoder(p)

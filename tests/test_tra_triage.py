import pytest
from scripts.run_tra_v2 import quick_decision

@pytest.mark.parametrize('successes,decision', [(0,'stop'), (5,'stop'), (6,'review_six'), (7,'full100'), (20,'full100')])
def test_quick20_preregistered_thresholds(successes, decision):
    assert quick_decision(successes, 20) == decision

@pytest.mark.parametrize('successes,valid,errors', [(7,19,0),(7,20,1),(21,20,0)])
def test_incomplete_or_invalid_quick_never_starts_full(successes, valid, errors):
    with pytest.raises(ValueError):
        quick_decision(successes, valid, errors)

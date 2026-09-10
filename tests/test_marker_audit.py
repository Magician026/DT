import numpy as np
import pytest
from scripts.audit_clean_v2 import validate_marker_pairs

def test_visibility_filter_changes_slots_but_preserves_displacements():
 # Generator filters both planes with the same indices, then repeats final pair.
 ref=np.array([[10.,20.],[30.,40.],[50.,60.]])
 frames=[]
 for ids in ([0,1,2],[0,2]):
  a=ref[ids];b=a+np.array([2.,-1.]);pair=np.stack([a,b]);frames.append(np.concatenate([pair,np.repeat(pair[:,-1:],1200-len(ids),axis=1)],axis=1))
 m=np.stack(frames)
 assert not np.allclose(m[:,0],m[0,0])
 report=validate_marker_pairs(m)
 assert report['reference_changes_across_frames']
 np.testing.assert_array_equal(m[:,1]-m[:,0],np.broadcast_to([2.,-1.],(2,1200,2)))

def test_invalid_marker_shape_and_nonfinite_rejected():
 with pytest.raises(ValueError):validate_marker_pairs(np.zeros((2,1200,2)))
 m=np.zeros((2,2,1200,2));m[0,1,0,0]=np.nan
 with pytest.raises(ValueError):validate_marker_pairs(m)

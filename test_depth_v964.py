import os,sys,importlib.util,unittest
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('depth_test',str(Path(__file__).resolve().parent/'auto_register_multiview1_v9.6.4_gpu.py'))
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
class DepthTests(unittest.TestCase):
 def setUp(self):self.model=dict(parameters=[1.006,-.008,.991],depth_normalizer=400.)
 def test_identity_and_no_input_mutation(self):
  P=np.array([[2.,3.,350.],[5.,-8.,420.]])
  original=P.copy();np.testing.assert_array_equal(m.depth_correct(P,dict(parameters=[1.,0.,1.],depth_normalizer=400.)),P)
  m.depth_correct(P,self.model);np.testing.assert_array_equal(P,original)
 def test_normals_follow_independent_finite_difference_jacobian(self):
  P=np.array([[2.,3.,350.],[5.,-8.,420.]]);N=np.array([[.2,.3,.9],[-.4,.8,.2]])
  actual=m.depth_correct_normals(P,N,self.model)
  for i in range(len(P)):
   J=np.column_stack([(m.depth_correct(P[i:i+1]+np.eye(3)[j]*1e-3,self.model)[0]-m.depth_correct(P[i:i+1]-np.eye(3)[j]*1e-3,self.model)[0])/2e-3 for j in range(3)])
   expected=np.linalg.solve(J.T,N[i]);expected/=np.linalg.norm(expected)
   np.testing.assert_allclose(actual[i],expected,atol=1e-10)
 def test_depth_changes_cache_key_and_reference_stays_original(self):
  path=str(Path(__file__).resolve());guide=str(Path(m.__file__).resolve())
  cfg=dict(m.DEFAULT_CFG,sphere_guide_path=guide)
  corrected=dict(cfg,_depth_model=self.model)
  self.assertNotEqual(m._prepared_key(path,cfg),m._prepared_key(path,corrected))
  self.assertEqual(m._prepared_key(guide,cfg),m._prepared_key(guide,corrected))
 def test_invalid_calibration_range_is_rejected(self):
  with self.assertRaises(ValueError):m.depth_correct(np.array([[1.,2.,400.]]),dict(parameters=[1.,-2.,1.],depth_normalizer=400.))
 def test_calibration_recovers_sensor_model_using_only_training_split(self):
  C=np.array([[0.,0.,0.],[52.,0.,2.],[21.,61.,1.],[-32.,22.,3.]])
  rng=np.random.default_rng(961);features={};a,b,k=self.model['parameters'];c=b/400.
  for frame in range(60):
   R=m.Rotation.from_rotvec(rng.normal(0,.25,3)).as_matrix()
   corrected=C@R.T+[rng.uniform(-20,20),rng.uniform(-20,20),rng.uniform(280,490)]
   denominator=a*k/(k-c*corrected[:,2]);raw=corrected*denominator[:,None]/[1.,1.,k]
   features[frame]=dict(match=dict(source_indices=list(range(4)),target_indices=list(range(4))),spheres=[dict(center=p) for p in raw])
  fitted=m._geometry_fit_depth(features,[dict(center=p) for p in C])
  np.testing.assert_allclose(fitted['parameters'],[a,b,k],atol=1e-7)
  self.assertLess(fitted['validation']['heldout']['after_rmse'],1e-7)
 def test_insufficient_distinct_calibration_views_are_rejected(self):
  with self.assertRaises(ValueError):m._geometry_fit_depth({},[dict(center=np.ones(3))]*4)
 def test_original_reference_read_and_input_scope(self):
  import tempfile
  with tempfile.TemporaryDirectory() as folder:
   source=Path(folder,'source.ply');reference=Path(folder,'reference.ply');P=np.array([[1.,2.,350.],[2.,4.,400.],[3.,1.,370.]])
   m.o3d.io.write_point_cloud(str(source),m._cloud_from_arrays(P));m.o3d.io.write_point_cloud(str(reference),m._cloud_from_arrays(P))
   before=m._DEPTH_MODEL,m._DEPTH_INPUTS
   try:
    m._DEPTH_MODEL=self.model;m._DEPTH_INPUTS={os.path.normcase(str(source.resolve()))}
    np.testing.assert_allclose(np.asarray(m.safe_read_point_cloud(str(source)).points),m.depth_correct(P,self.model))
    np.testing.assert_array_equal(np.asarray(m.safe_read_point_cloud(str(reference)).points),P)
    np.testing.assert_array_equal(np.asarray(m._DEPTH_BASE_READ(str(source)).points),P)
   finally:m._DEPTH_MODEL,m._DEPTH_INPUTS=before
if __name__=='__main__':unittest.main(verbosity=2)



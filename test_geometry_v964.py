import os,sys,unittest,importlib.util
from pathlib import Path
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import numpy as np
from scipy.spatial.transform import Rotation
spec=importlib.util.spec_from_file_location('geometry_test',str(Path(__file__).resolve().parent/'auto_register_multiview1_v9.6.4_gpu.py'))
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)


class GeometryTests(unittest.TestCase):
 def setUp(self):
  self.centers=np.array([[0.,0.,0.],[52.,0.,2.],[21.,61.,1.],[-32.,22.,3.]])
  self.radii=np.array([5.7,5.8,5.75,5.78])
  self.rotation=Rotation.from_rotvec([.35,-.23,.41]).as_matrix()
  self.translation=-self.rotation@np.array([0.,0.,350.])
  self.T=np.eye(4);self.T[:3,:3]=self.rotation;self.T[:3,3]=self.translation
  self.source_centers=(self.centers-self.translation)@self.rotation
  self.target=[dict(center=c,radius=r,n=1000,rmse=.02,condition=80,votes=100) for c,r in zip(self.centers,self.radii)]
 def test_unlabelled_geometry_recovers_rigid_pose_with_false_candidate(self):
  permutation=[2,0,3,1]
  source=[dict(self.target[i],center=self.source_centers[i]) for i in permutation]
  source.append(dict(self.target[0],center=np.array([210.,-110.,500.])))
  result=m.match_three_or_four(source,self.target,.1)
  self.assertIsNotNone(result)
  np.testing.assert_allclose(result['T'],self.T,atol=1e-10)
 def test_three_noncollinear_markers_recover_visible_subset(self):
  source=[dict(self.target[i],center=self.source_centers[i]) for i in [3,1,0]]
  result=m.match_three_or_four(source,self.target,.1)
  self.assertIsNotNone(result);np.testing.assert_allclose(result['T'],self.T,atol=1e-10)
 def test_two_markers_do_not_falsely_claim_complete_pose(self):
  source=[dict(self.target[i],center=self.source_centers[i]) for i in [0,1]]
  self.assertIsNone(m.match_three_or_four(source,self.target,.1))
 def test_angular_search_covers_remaining_degree_of_freedom(self):
  source=[dict(self.target[i],center=self.source_centers[i]) for i in [0,1]]
  transforms=m.two_sphere_transforms(source,self.target,.1,angle_step=6.)
  self.assertGreaterEqual(len(transforms),120)
  errors=[Rotation.from_matrix(c['T'][:3,:3]@self.rotation.T).magnitude() for c in transforms]
  self.assertLessEqual(min(errors),np.deg2rad(3.)+1e-9)
  for candidate in transforms:
   self.assertAlmostEqual(np.linalg.det(candidate['T'][:3,:3]),1.,places=12)
 def _sample(self,count=4):
  rng=np.random.default_rng(620);points=[];normals=[]
  for c,r in zip(self.centers[:count],self.radii[:count]):
   n=rng.normal(size=(800,3));n[:,2]=-abs(n[:,2])-.5;n/=np.linalg.norm(n,axis=1)[:,None]
   points.append(c+r*n+rng.normal(scale=.002,size=n.shape));normals.append(n)
  world=np.vstack(points);N=np.vstack(normals)
  return (world-self.translation)@self.rotation,N@self.rotation
 def test_analytic_refinement_reduces_pose_error_and_preserves_distances(self):
  P,N=self._sample();before=P.copy()
  correction=np.eye(4);correction[:3,:3]=Rotation.from_rotvec([.001,-.002,.0015]).as_matrix()
  pivot=self.centers.mean(0);correction[:3,3]=pivot+np.array([.08,-.04,.05])-correction[:3,:3]@pivot
  initial=correction@self.T
  actual,report=m.refine_sphere_pose(P,N,initial,self.centers,self.radii)
  self.assertTrue(report['accepted']);self.assertLess(report['final_rmse'],report['initial_rmse']*.1)
  expected=P@self.rotation.T+self.translation;result=P@actual[:3,:3].T+actual[:3,3]
  self.assertLess(np.sqrt(np.mean(np.sum((result-expected)**2,axis=1))),.02)
  np.testing.assert_array_equal(P,before)
  np.testing.assert_allclose(np.linalg.norm(result[1:100]-result[:99],axis=1),np.linalg.norm(P[1:100]-P[:99],axis=1),atol=1e-10)
 def test_partial_geometry_keeps_initial_pose_when_unobservable(self):
  P,N=self._sample(2);initial=self.T.copy()
  actual,report=m.refine_sphere_pose(P,N,initial,self.centers,self.radii)
  self.assertFalse(report['accepted']);np.testing.assert_array_equal(actual,initial)
 def test_free_sphere_fit_does_not_force_detection_radius(self):
  rng=np.random.default_rng(19);n=rng.normal(size=(2000,3));n/=np.linalg.norm(n,axis=1)[:,None]
  radius=6.12;center=np.array([20.,-11.,350.]);P=center+radius*n
  fit=m.fit_sphere(P,center+[.1,-.2,.05],initial_radius=5.8)
  self.assertIsNotNone(fit);np.testing.assert_allclose(fit['center'],center,atol=1e-8)
  self.assertAlmostEqual(fit['radius'],radius,places=8)


if __name__=='__main__':unittest.main(verbosity=2)




import os,sys,unittest,importlib.util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[v]='1'
base=Path(__file__).resolve().parent
os.environ.setdefault('CUPY_CACHE_DIR',str(base/'.cupy_cache'))
import numpy as np
from scipy.spatial import cKDTree
spec=importlib.util.spec_from_file_location('reg_gpu_tests',base/'auto_register_multiview1_v9.6.4_gpu.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
SharedExactCuda=m.SharedExactCuda
cp=m._gpu_backend().cp
class SharedGpuTests(unittest.TestCase):
 def test_normals_added_after_point_to_point_cache(self):
  import open3d as o3d
  g=SharedExactCuda();rng=np.random.default_rng(1)
  cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rng.normal(size=(500,3))))
  cached=g.cloud(cloud,tree=True)
  self.assertIsNone(cached['n'])
  cloud.estimate_normals()
  updated=g.cloud(cloud,normals=True,tree=True)
  np.testing.assert_array_equal(cp.asnumpy(updated['n']),np.asarray(cloud.normals))
 def test_hints_are_only_exact_upper_bounds(self):
  g=SharedExactCuda();rng=np.random.default_rng(83)
  target=rng.normal(size=(17000,3));query=rng.normal(size=(2000,3))
  for hints in [rng.integers(0,len(target)+2,len(query),dtype=np.int64),cKDTree(target).query(query+.02)[1]]:
   d,ix=g.query_hint(target,query,.7,cp.asarray(hints))
   exact=cKDTree(target).query(query,distance_upper_bound=.7)[0]
   np.testing.assert_allclose(cp.asnumpy(d),exact,atol=1e-12)
 def test_labelled_search_excludes_own_frame(self):
  g=SharedExactCuda();rng=np.random.default_rng(5)
  class Owner:pass
  obj=Owner();obj.points=rng.normal(size=(12000,3))
  obj.frame_labels=np.repeat(np.arange(4,dtype=np.int32),3000)
  actual=cp.asnumpy(g.query_excluding_frames(obj))
  expected=np.concatenate([cKDTree(obj.points[obj.frame_labels!=k]).query(obj.points[obj.frame_labels==k])[0] for k in range(4)])
  np.testing.assert_allclose(actual,expected,atol=1e-12)
  single=Owner();single.points=obj.points;single.frame_labels=np.zeros(12000,np.int32)
  self.assertTrue(np.isinf(cp.asnumpy(g.query_excluding_frames(single))).all())
 def test_exact_all_tree_sizes_and_degenerate_clouds(self):
  g=SharedExactCuda();rng=np.random.default_rng(154)
  for n in [1,2,3,7,16,17,31,33,100,1025,5000]:
   for flat in [False,True]:
    target=rng.normal(size=(n,3))
    if flat:target[:,2]=0.
    query=rng.normal(size=(70,3))
    for bound in [np.inf,.3]:
     expected=cKDTree(target).query(query,distance_upper_bound=bound)[0]
     d,ix=g.query(target,query,bound);actual=cp.asnumpy(d);indices=cp.asnumpy(ix)
     np.testing.assert_array_equal(np.isfinite(actual),np.isfinite(expected))
     valid=np.isfinite(expected)
     np.testing.assert_allclose(actual[valid],expected[valid],rtol=1e-11,atol=1e-11)
     np.testing.assert_allclose(np.linalg.norm(query[valid]-target[indices[valid]],axis=1),expected[valid],rtol=1e-11,atol=1e-11)
 def test_strict_distance_boundary(self):
  g=SharedExactCuda();target=np.array([[0.,0.,0.],[2.,0.,0.]])
  query=np.array([[1.,0.,0.],[.5,0.,0.]])
  d,ix=g.query(target,query,1.)
  np.testing.assert_array_equal(np.isfinite(cp.asnumpy(d)),[False,True])
 def test_concurrent_shared_tree_upload_and_queries(self):
  g=SharedExactCuda();rng=np.random.default_rng(6);target=rng.normal(size=(30000,3));query=target[:4000]+.003
  expected=cKDTree(target).query(query)[0]
  def worker(_):
   cp.cuda.Device(0).use()
   with cp.cuda.Stream(non_blocking=True):
    d,ix=g.query(target,query)
    np.testing.assert_allclose(cp.asnumpy(d),expected,atol=1e-11,rtol=1e-11)
  with ThreadPoolExecutor(4) as pool:list(pool.map(worker,range(8)))



class PostprocessTests(unittest.TestCase):
 def test_point_to_point_matches_cpu(self):
  rng=np.random.default_rng(73);P=rng.uniform(-10,10,(7000,3));P[:,2]=.03*P[:,0]**2+.5*np.sin(P[:,1])
  R=m.Rotation.from_rotvec([.002,-.003,.004]).as_matrix();Q=P@R.T+[.03,-.02,.01]
  source,target=m._cloud_from_arrays(P),m._cloud_from_arrays(Q)
  cpu=m._POST_CPU_PP(source,target,.8,.8,40)
  cuda=m.icp_pp(source,target,.8,.8,40)
  displacement=np.linalg.norm(P@(cpu[:3,:3]-cuda[:3,:3]).T+cpu[:3,3]-cuda[:3,3],axis=1)
  self.assertLess(displacement.max(),1e-8)
  # Point-to-point cached these clouds before normals existed.
  cpu,_=m._CPU_ICP(source,target,.5,.7,30,T_init=cpu)
  cuda,_=m.icp_custom(source,target,.5,.7,30,T_init=cuda)
  self.assertLess(np.linalg.norm(P@(cpu[:3,:3]-cuda[:3,:3]).T+cpu[:3,3]-cuda[:3,3],axis=1).max(),1e-8)
 def test_frame_support_matches_original_and_boundary(self):
  rng=np.random.default_rng(12);P=rng.uniform(-3,3,(500,3))
  raws=[m._cloud_from_arrays(P+rng.normal(scale=.03,size=P.shape)) for _ in range(3)]
  poses={k:np.eye(4) for k in range(3)}
  cfg=dict(m.DEFAULT_CFG,score_voxel=.1,merge_voxel=.1,cover_radius=.15,cover_min=2)
  obj=m.Merger(cfg)
  expected=m._POST_CPU_MERGE(obj,raws,poses,[0,1,2])[1]
  actual=obj.merge_and_clean(raws,poses,[0,1,2])[1]
  a,b=np.asarray(actual.points),np.asarray(expected.points)
  self.assertEqual(len(a),len(b));self.assertLess(cKDTree(a).query(b)[0].max(),1e-12)
 def test_other_frame_scores_match_cpu_with_isolated_patches(self):
  rng=np.random.default_rng(18)
  world={0:rng.normal(size=(4000,3)),1:rng.normal(size=(4000,3))+.1,
         2:rng.normal(scale=.1,size=(3000,3))+[25.,0.,0.]}
  cpu=m._POST_CPU_CROSS(world);cuda=m._cross_frame_distances(world)
  for k in world:np.testing.assert_allclose(cpu[k],cuda[k],rtol=1e-11,atol=1e-11)




class GPUVerification(unittest.TestCase):
 def test_cutoff_preserves_even_and_odd_median_decisions(self):
  bound=m._pair_score_bound(m.DEFAULT_CFG)
  cases=[[0.,100.],[0.,4.],[1.,5.],[0.,.2,100.],[1.,2.,4.,100.]]
  rng=np.random.default_rng(29)
  cases.extend(rng.uniform(0,30,size=n) for n in range(1,100))
  for values in cases:
   true=np.median(values);clipped=np.median(np.minimum(values,bound))
   for limit in [.02,.4,.6,1.,1.2,2.]:self.assertEqual(true<=limit,clipped<=limit)
   if true<=2.:self.assertEqual(true,clipped)

 def test_exact_gpu_neighbours_with_large_origin_duplicates_and_bounds(self):
  gpu=m.ExactCudaBackend()
  rng=np.random.default_rng(18)
  points=rng.normal(size=(5000,3))+[100000.,-200000.,300000.]
  points=np.vstack((points,points[:20]))
  query=np.vstack((points[:400]+rng.normal(scale=.02,size=(400,3)),points[:20]+[100.,100.,100.]))
  for radius in [np.inf,.1,4.000001]:
   expected,ix=cKDTree(points).query(query,distance_upper_bound=radius)
   d,index=gpu.query(points,query,bound=radius)
   actual=gpu.cp.asnumpy(d);indices=gpu.cp.asnumpy(index)
   np.testing.assert_array_equal(np.isfinite(actual),np.isfinite(expected))
   take=np.isfinite(expected)
   np.testing.assert_allclose(actual[take],expected[take],rtol=1e-10,atol=1e-10)
   distances=np.linalg.norm(query[take]-points[indices[take]],axis=1)
   np.testing.assert_allclose(distances,expected[take],rtol=1e-10,atol=1e-10)

 def test_degenerate_plane_no_drift_cpu_gpu_agree(self):
  rng=np.random.default_rng(2)
  q=np.column_stack((rng.normal(size=(500,2)),np.zeros(500)))+[1000.,-300.,900.]
  p=q+[0.,0.,.2];normals=np.tile([0.,0.,1.],(len(q),1))
  gpu=m.ExactCudaBackend();cp=gpu.cp
  a=m.p2plane_step(p,q,normals)
  b=gpu.plane_step(cp.asarray(p),cp.asarray(q),cp.asarray(normals))
  np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-10)
  np.testing.assert_allclose(b[:2,3],0.,atol=1e-10)


if __name__=='__main__':unittest.main(verbosity=2)


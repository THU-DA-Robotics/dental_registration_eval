import contextlib,importlib.util,io,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
import numpy as np
spec=importlib.util.spec_from_file_location('manual_bridge_test',Path(__file__).resolve().parent/'auto_register_multiview1_v9.6.4_gpu.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

class ManualBridgeTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.out=Path(self.tmp.name)
  rng=np.random.default_rng(964);self.P=rng.normal(size=(180,3))*4+[0,0,350]
  self.raws=[m._cloud_from_arrays(self.P+[k,0,0]) for k in range(5)]
  self.edge_T=np.eye(4);self.edge_T[:3,:3]=m.Rotation.from_rotvec([.05,-.03,.02]).as_matrix();self.edge_T[:3,3]=[2.,-1.,.3]
  self.edge=m.PairEdge(2,3,self.edge_T,np.eye(6),.05,.9,True,True)
  self.confirmed=np.eye(4);self.confirmed[:3,:3]=m.Rotation.from_rotvec([.2,.1,-.15]).as_matrix();self.confirmed[:3,3]=[8.,-5.,3.]
 def config(self,manual):return dict(m.DEFAULT_CFG,manual_rescue=manual,graph_refine=False,freeze_seed_main=True,rescue_rounds=3,_outdir=str(self.out))
 def rescue(self,manual,*,undo=False,polish=False):
  cfg=self.config(manual)
  if polish:cfg['freeze_seed_main']=False
  raws=self.raws if undo else self.raws[:4]
  def wrong_polish(raws,use,poses):
   for T in poses.values():T[:3,3]+=100.
   return poses
  with patch.object(m,'_rescue_align_fast',return_value=None),patch.object(m.ManualAligner,'manual_align',return_value=self.confirmed) as picker,patch('builtins.input',side_effect=['y','u'] if undo else ['y']) as ask,patch.object(m.GlobalOptimizer,'_multiway_icp',side_effect=wrong_polish),contextlib.redirect_stdout(io.StringIO()):
   poses,use,_=m.rescue_failed_frames(raws,{0:np.eye(4),1:np.eye(4)},[0,1],cfg,edges=[self.edge])
  return cfg,poses,use,picker,ask
 def test_default_is_automatic_and_never_prompts(self):
  self.assertFalse(m.DEFAULT_CFG['manual_rescue'])
  cfg,poses,use,picker,ask=self.rescue(False)
  self.assertEqual(use,[0,1]);picker.assert_not_called();ask.assert_not_called()
 def test_manual_component_keeps_internal_relative_pose_and_initial_main(self):
  cfg,poses,use,picker,ask=self.rescue(True)
  self.assertEqual(use,[0,1,2,3]);self.assertEqual(cfg['_manual_bridge_frames'],[2,3]);picker.assert_called_once()
  np.testing.assert_allclose(np.linalg.inv(poses[2])@poses[3],np.linalg.inv(self.edge_T),atol=1e-12)
  np.testing.assert_array_equal(poses[2],self.confirmed)
  for k in [0,1]:np.testing.assert_array_equal(poses[k],np.eye(4))
 def test_manual_component_is_restored_after_optional_polish(self):
  cfg,poses,use,_,_=self.rescue(True,polish=True)
  np.testing.assert_array_equal(poses[2],self.confirmed)
  np.testing.assert_allclose(np.linalg.inv(poses[2])@poses[3],np.linalg.inv(self.edge_T),atol=1e-12)
 def test_undo_removes_confirmed_pose_locks(self):
  cfg,poses,use,_,_=self.rescue(True,undo=True)
  self.assertEqual(use,[0,1]);self.assertEqual(cfg['_manual_bridge_frames'],[])
 def test_manual_locks_skip_later_sphere_and_surface_refinement(self):
  cfg,poses,use,_,_=self.rescue(True);manual={k:poses[k] for k in [2,3]}
  cfg['_files']=['missing']*4
  result,checks=m._geometry_refine_many(['missing']*4,manual,cfg,[dict(center=[0,0,0],radius=5.8)])
  fake_gpu=Mock();fake_gpu.cp=None
  with patch.object(m,'_gpu_backend',return_value=fake_gpu):
   surface=m._geometry_surface_refine(['missing']*4,result,cfg,None)
  for k in manual:
   np.testing.assert_array_equal(surface[k],manual[k]);self.assertEqual(checks[k]['reason'],'manual_bridge_locked')
 def test_final_manual_audit_uses_exported_coordinate_frame(self):
  cfg,poses,use,_,_=self.rescue(True)
  G=np.eye(4);G[:3,3]=[100.,200.,300.];exported={k:G@T for k,T in poses.items()}
  with patch.object(m,'_DEPTH_BASE_SAVE'):
   m.save_outputs(str(self.out),None,None,exported,use,{},0,cfg)
  audit=json.loads((self.out/'manual_bridge_audit.json').read_text(encoding='utf8'))
  self.assertEqual(audit['confirmed_frames'],[2,3]);np.testing.assert_array_equal(audit['confirmed_final_poses']['2'],exported[2])
 def cli_config(self,flags):
  file=self.out/'input.ply';file.write_bytes(b'placeholder');captured={}
  with patch.object(sys,'argv',['registration',str(self.out),'--precheck-only','--backend','cpu','--output-dir',str(self.out/'result'),*flags]),patch.object(m,'_validate_cuda_startup',side_effect=lambda cfg:captured.update(cfg)),patch.object(m,'precheck_point_cloud_files',return_value=[]),contextlib.redirect_stdout(io.StringIO()):m.main()
  return captured
 def test_cli_manual_flag_overrides_generic_set(self):
  cfg=self.cli_config(['--manual-bridge','--set','manual_rescue=false'])
  self.assertTrue(cfg['manual_rescue']);self.assertEqual(cfg['_manual_bridge_frames'],[])
 def test_cli_disable_alias_and_default(self):
  self.assertFalse(self.cli_config([])['manual_rescue'])
  self.assertFalse(self.cli_config(['--no-manual-rescue'])['manual_rescue'])
 def test_conflicting_manual_flags_fail_before_registration(self):
  with patch.object(sys,'argv',['registration','input','--manual-bridge','--no-manual-rescue']),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as e:m.main()
  self.assertEqual(e.exception.code,2)
 def test_picking_point_cloud_conversion_without_opening_window(self):
  poly=m.ManualAligner._o3d_to_pv(self.raws[0]);self.assertEqual(poly.n_points,len(self.P))

if __name__=='__main__':unittest.main(verbosity=2)

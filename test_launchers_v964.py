import json,os,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path

@unittest.skipUnless(os.name=='nt','Windows CMD launchers')
class LauncherTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.folder=Path(self.temp.name)
  base=Path(__file__).resolve().parent
  for name in ['_gpu_environment.cmd','run_gpu_no_reference.cmd','run_gpu_calibrated.cmd']:shutil.copy2(base/name,self.folder/name)
  (self.folder/'auto_register_multiview1_v9.6.4_gpu.py').write_text("import os,sys,json\nfrom pathlib import Path\nPath(os.environ['ARG_CAPTURE']).write_text(json.dumps(sys.argv[1:]),encoding='utf8')\n",encoding='utf8')
  self.data=self.folder/'object with spaces';self.data.mkdir();self.output=self.folder/'result with spaces'
  self.capture=self.folder/'args.json';self.env=dict(os.environ,REG_PYTHON=sys.executable,ARG_CAPTURE=str(self.capture))
 def invoke(self,name,args):
  command=subprocess.list2cmdline([str(self.folder/name),*[str(v) for v in args]])
  # CMD needs its command string quoted once, without Python's argv re-escaping.
  shell=os.environ.get('COMSPEC','cmd.exe')
  result=subprocess.run(f'"{shell}" /d /s /c "{command}"',env=self.env,capture_output=True)
  actual=json.loads(self.capture.read_text(encoding='utf8')) if self.capture.exists() else None
  return result,actual
 def test_no_reference_forwards_manual_and_check_with_spaced_paths(self):
  result,args=self.invoke('run_gpu_no_reference.cmd',[self.data,self.output,'--manual-bridge','--check'])
  self.assertEqual(result.returncode,0,result.stdout+result.stderr);self.assertIn('--manual-bridge',args);self.assertNotIn('--no-manual-rescue',args)
  self.assertIn('--precheck-only',args);self.assertEqual(Path(args[0]),self.data);self.assertEqual(Path(args[args.index('--output-dir')+1]),self.output)
 def test_no_reference_keeps_automatic_default(self):
  result,args=self.invoke('run_gpu_no_reference.cmd',[self.data,self.output])
  self.assertEqual(result.returncode,0);self.assertIn('--no-manual-rescue',args);self.assertNotIn('--manual-bridge',args)
 def test_calibrated_forwards_optional_flags_after_four_paths(self):
  guide=self.folder/'sphere guide.ply';surface=self.folder/'surface guide.ply';guide.touch();surface.touch()
  result,args=self.invoke('run_gpu_calibrated.cmd',[self.data,guide,surface,self.output,'--manual-bridge','--check'])
  self.assertEqual(result.returncode,0,result.stdout+result.stderr);self.assertIn('--manual-bridge',args);self.assertNotIn('--no-manual-rescue',args)
  self.assertIn('--precheck-only',args);self.assertEqual(Path(args[args.index('--sphere-guide')+1]),guide);self.assertEqual(Path(args[args.index('--surface-reference')+1]),surface)
 def test_unknown_option_fails_without_launching_python(self):
  result,args=self.invoke('run_gpu_no_reference.cmd',[self.data,self.output,'--unknown-option'])
  self.assertEqual(result.returncode,1);self.assertIsNone(args)

if __name__=='__main__':unittest.main(verbosity=2)

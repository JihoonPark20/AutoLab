import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rl_autolab import parse_proposal, read_json, init_state, accept_result, check_agent_budget, validate_config


class TestAutolab(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=read_json(ROOT/'configs/demo.json')

    def test_reject_unsafe_changes(self):
        base={k:v['value'] for k,v in self.cfg['parameters'].items()}
        for updates in [{'not_allowlisted':.2}, {'curriculum_difficulty':100},
                        {'curriculum_difficulty':.8}, {'reward_tracking':'rm -rf /'},
                        {'reward_tracking':float('nan')}]:
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                parse_proposal({'updates':updates},base,self.cfg)

    def test_valid_proposal(self):
        base={k:v['value'] for k,v in self.cfg['parameters'].items()}
        got=parse_proposal({'updates':{'curriculum_difficulty':.30}},base,self.cfg)
        self.assertEqual(got['curriculum_difficulty'],.30)
        self.assertEqual(got['reward_tracking'],1.0)

    def test_heldout_guardrail(self):
        st=init_state(self.cfg)
        res={'status':'ok','metrics':{'success_rate':.91,'fall_rate':.7,'energy':1.2}}
        self.assertFalse(accept_result(st,res,self.cfg,{})[0])

    def test_budget(self):
        cfg=copy.deepcopy(self.cfg)
        cfg['backend']='claude'
        st=init_state(cfg)
        st['agent_calls']=[__import__('time').time() for _ in range(5)]
        self.assertFalse(check_agent_budget(st,cfg)[0])

    def test_end_to_end_cpu_mock(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            cfg=copy.deepcopy(self.cfg)
            cfg['train_seeds']=[40,41]
            cfg['decision']['target_score']=.75
            config_path=td/'cfg.json'
            config_path.write_text(json.dumps(cfg))
            state_dir=td/'state'
            cmd=[sys.executable,str(ROOT/'rl_autolab.py'),'run','--config',str(config_path),
                 '--state-dir',str(state_dir),'--cycles','2']
            got=subprocess.run(cmd,text=True,capture_output=True,timeout=35)
            self.assertEqual(got.returncode,0,got.stdout+'\n'+got.stderr)
            st=read_json(state_dir/'state.json')
            self.assertEqual(st['next_cycle'],2)
            self.assertEqual(len(st['history']),2)
            self.assertTrue(st['history'][1]['accepted'])
            result=read_json(state_dir/'runs/cycle_001/worker_result.json')
            self.assertEqual(len(result['per_seed']),2)
            for seed in [40,41]:
                evaluation=read_json(state_dir/f'runs/cycle_001/seed_{seed}/eval.json')
                self.assertEqual(evaluation['evaluation_seeds'], cfg['eval']['fixed_seeds'])

    def test_manual_pause_and_proposal(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            cfg=copy.deepcopy(self.cfg)
            cfg['backend']='manual'
            config_path=td/'cfg.json'
            config_path.write_text(json.dumps(cfg))
            state_dir=td/'state'
            cmd=[sys.executable,str(ROOT/'rl_autolab.py'),'run','--config',str(config_path),
                 '--state-dir',str(state_dir),'--cycles','2']
            got=subprocess.run(cmd,text=True,capture_output=True,timeout=30)
            self.assertEqual(got.returncode,3,got.stdout+'\n'+got.stderr)
            self.assertTrue((state_dir/'proposal_request.md').exists())
            (state_dir/'proposal.json').write_text(json.dumps({'updates':{'curriculum_difficulty':.3}}))
            got=subprocess.run(cmd,text=True,capture_output=True,timeout=30)
            self.assertEqual(got.returncode,0,got.stdout+'\n'+got.stderr)
            self.assertEqual(read_json(state_dir/'state.json')['next_cycle'],2)

    def test_finetune_from_champion(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            cfg=copy.deepcopy(self.cfg)
            cfg['training_mode']='finetune_from_champion'
            cfg['train_seeds']=[40]
            config_path=td/'cfg.json'
            config_path.write_text(json.dumps(cfg))
            state_dir=td/'state'
            cmd=[sys.executable,str(ROOT/'rl_autolab.py'),'run','--config',str(config_path),
                 '--state-dir',str(state_dir),'--cycles','2']
            got=subprocess.run(cmd,text=True,capture_output=True,timeout=35)
            self.assertEqual(got.returncode,0,got.stdout+'\n'+got.stderr)
            a=read_json(state_dir/'runs/cycle_000/seed_40/checkpoint.mock')
            b=read_json(state_dir/'runs/cycle_001/seed_40/checkpoint.mock')
            self.assertEqual(a['resumed_from'],'')
            self.assertEqual(b['resumed_from'],str((state_dir/'runs/cycle_000/seed_40/checkpoint.mock').resolve()))

    def test_code_worktree_separation(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            repo=td/'research'; repo.mkdir()
            def git(*args):
                return subprocess.run(['git','-C',str(repo),*args],text=True,capture_output=True)
            self.assertEqual(git('init').returncode,0)
            (repo/'hello.py').write_text('print(1)\n')
            self.assertEqual(git('add','hello.py').returncode,0)
            self.assertEqual(git('-c','user.name=test','-c','user.email=test@example.com','commit','-m','init').returncode,0)
            state=init_state(self.cfg)
            state['history']=[{'cycle':0,'metrics':{'success_rate':0.1}}]
            state_dir=td/'state';state_dir.mkdir()
            (state_dir/'state.json').write_text(json.dumps(state))
            worktree=td/'mdp-worktree'
            got=subprocess.run([sys.executable,str(ROOT/'scripts/code_lane.py'),
                               '--repo',str(repo),'--state-dir',str(state_dir),
                               '--branch','autolab/test-review','--worktree',str(worktree),
                               '--task','Improve reset distribution'],
                               text=True,capture_output=True)
            self.assertEqual(got.returncode,0,got.stdout+'\n'+got.stderr)
            self.assertTrue((worktree/'AUTOLAB_CODE_TASK.md').is_file())
            self.assertFalse((repo/'AUTOLAB_CODE_TASK.md').exists())


if __name__=='__main__':unittest.main()

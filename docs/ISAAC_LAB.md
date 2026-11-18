# Isaac Lab 2.3.x + RSL-RL 구체적 연결 계약

환경 설계가 미정이므로 등록 Task ID / RewardCfg 클래스 / rollout 성공 판정은 템플릿 값입니다. **이 세 가지를 알지 못한 상태에서 실제 평가 성공률을 계산할 수 없으므로, 채워야 하는 연결 지점을 명시합니다.**

## A. 준비 확인

```bash
export ISAACLAB_ROOT="$HOME/IsaacLab"
export ISAAC_PYTHON="$HOME/miniconda3/envs/isaaclab232/bin/python"
"$ISAAC_PYTHON" "$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/train.py" --help
"$ISAAC_PYTHON" -c 'import isaaclab; print("isaaclab ready")'
```

- 실제 연구 repo가 Isaac Lab checkout 밖에 있으면, 이 Python 환경에 연구 Task package를 설치하거나 PYTHONPATH에 추가해야 Task ID가 등록됩니다.
- 2.3.2 기준 RSL-RL 학습 스크립트 경로 `scripts/reinforcement_learning/rsl_rl/train.py`; 3.x는 CLI/스크립트 구조가 달라 이 어댑터를 그대로 쓸 수 없습니다.

## B. mutable MDP 변수 2개를 기존 Config에 노출

예시 `configs/my_hydra_map.json`:

```json
{
  "curriculum_difficulty": "env.curriculum.terrain_levels.params.difficulty",
  "reward_tracking": "env.rewards.track_lin_vel_xy_exp.weight"
}
```

**주의:** 위 curriculum term은 예시일 뿐입니다. `env.curriculum.terrain_levels.params.difficulty`가 실제 사용자 Task config에 있는지 확인해야 합니다. 없으면 Config 클래스에 scalar를 명시적으로 정의하거나, 실제로 존재하는 파라미터만 자동 변경 대상으로 정하세요. `env`로 시작하는 Hydra 경로만 허용하도록 제한되어 있습니다.

`configs/my_project.json`에서 

1. `train.command`의 `YOUR_REGISTERED_TASK_ID`를 등록된 Task ID로 변경.
2. `--map-json configs/hydra_map.example.json`을 `--map-json configs/my_hydra_map.json`으로 변경.
3. `--num-envs 4096`, `--max-iterations 10000`을 GPU·실험 설정에 맞게 설정.
4. `train_seeds: [40,41,42]`와 `eval.fixed_seeds: [501,...]`를 서로 겹치지 않도록 고정.
5. `train.timeout_seconds`, `eval.timeout_seconds`를 최악 실행 시간보다 길게 설정.
6. `backend=manual`부터 시작해 실제 자동화를 검증한 후 `backend=claude`로 전환.

학습 커맨드는 내부에서 이런 형태로 확장됩니다.

```bash
"$ISAAC_PYTHON" \
  "$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/train.py" \
  --task=YOUR_REGISTERED_TASK_ID --headless --num_envs=4096 \
  --max_iterations=10000 --seed=40 \
  agent.experiment_name=rl_autolab \
  agent.run_name=<cycle-and-seed-unique-id> \
  env.rewards.track_lin_vel_xy_exp.weight=1.0 \
  env.curriculum.terrain_levels.params.difficulty=0.2
```

학습 완료 시 `logs/rsl_rl/rl_autolab/<timestamp>_<unique-name>/model_*.pt`를 찾아 해당 seed run 디렉터리에 `checkpoint_path.txt`를 생성합니다. 이 파일이 없으면 평가를 시작하지 않고 실패 처리합니다.

**Checkpoint 이어 학습:** `training_mode=finetune_from_champion`인 경우, 같은 training seed의 챔피언 checkpoint 위치를 참조해 `agent.resume=true`, `agent.load_run=<folder>`, `agent.load_checkpoint=<model_N.pt>`를 추가합니다. 관측 공간/action 차원이 변경된 코드 변경에는 그대로 쓸 수 없으므로 cold-start 재학습을 권장합니다.

## C. 독립 평가 Adapter 연결 — 반드시 실제 구현

필요한 입력(Manager가 제공):

```bash
python scripts/isaac_eval_contract.py \
  --task YOUR_REGISTERED_TASK_ID \
  --checkpoint /absolute/path/model_10000.pt \
  --fixed-seeds 501,502,503,504,505 \
  --run-dir /absolute/path/seed_40
```

연구 코드에 `evaluate_policy(checkpoint, task, seeds)` 함수를 구현하고, `export RL_EVAL_FUNCTION=your_project.eval:evaluate_policy`로 연결합니다. 이 환경변수가 없으면 평가 adapter는 임의 값을 채우지 않고 오류를 발생시킵니다. 구현 방식의 예:

- Isaac Lab `AppLauncher`를 headless로 실행.
- `gym.make(task, cfg=env_cfg)`로 **변경하지 않는 평가 전용** Task 생성.
- RSL-RL `OnPolicyRunner` / `RslRlVecEnvWrapper`를 통해 checkpoint의 policy weight 로딩.
- 각 seed에서 정해진 에피소드 수만큼 action rollout; 성공/실패/낙상/에너지 지표를 Task-specific getter에서 읽기.
- 성공 이벤트의 정의는 reset reason/termination term과 별도로 명시. Reward sum을 임의로 성공률로 환산하지 않기.
- 반드시 `evaluation_seeds`와 수치 `metrics`를 다음과 같이 출력:

```json
{
  "evaluation_seeds": [501, 502, 503, 504, 505],
  "metrics": {
    "success_rate": 0.83,
    "fall_rate": 0.07,
    "energy": 1.17,
    "worst_seed_success": 0.74
  }
}
```

위 숫자는 **포맷을 보여주는 예시**이며 실험 결과가 아닙니다. 실제 rollout을 계산해야 합니다. `energy`의 정의 (예: `|τ·qdot|` 시간 적분을 이동거리로 정규화)를 연구 목적에 따라 고정하고 guardrail에도 같은 단위를 사용하세요.

### 왜 평가를 자동 코드 생성하지 않았나?

예를 들어 foot-slip rate, user exclusion zone, torque exposure, CoP violation, touchdown impact는 Task마다 episode-level 정의와 접근 방식이 완전히 다릅니다. 성공률을 `reward`로 근사하면 AI가 reward 설계를 변경할 때 평가를 속일 수 있습니다. **평가기 오염 방지**를 위해 실제 환경에서 반환한 메트릭만 인정하도록 했습니다.

## D. 학습 중단과 curriculum의 경계

1. **학습 내부 자동 curriculum:** RL 코드 내부의 성취도 기반 difficulty 변경. 토큰 소비 없음. MDP 변화 이력 로깅 필수.
2. **외부 AI intervention:** 구조적인 plateau/failure 관측 후 checkpoint 저장, 학습 종료, 원인 분석, 새 training distribution으로 다음 실험. AI 1회 호출.
3. **짧은 중간 점검:** training hook이 매 50 iterations `train_metrics.jsonl`에 summary를 기록. `early_stop.enabled=true`로 지정하면 plateau 조건과 checkpoint 존재 여부를 확인하여 조기 중단. 실제 환경이 로그를 생성하지 않으면 이 기능을 활성화하지 마세요.

로그 예시(한 줄에 JSON 하나):

```json
{"iteration": 50, "train_success": 0.34, "loss_policy": 0.18, "value_loss": 0.05}
```

**정지 후 재학습**과 **학습 프로세스 안에서 MDP 변경**은 다릅니다. 안전하고 재현 가능한 첫 버전은 위 (2)처럼 checkpoint 경계에서 분포를 변경합니다. 진짜 학습 중 즉시 개입하려면 RL worker와 제어 프로세스 사이에 IPC/config reload hook, PPO rollout boundary에서 파라미터 적용, curriculum version 기록까지 추가해야 합니다.

## E. 과학적 결과와 실험 승인

- 간단한 `min_improvement=0.02`는 효과 크기 필터입니다. 통계 검정 아님.
- 탐색 중 고정 validation seed를 반복 사용하는 것은 그 분포에 대한 과적합 위험을 가짐. 최종 논문용 unseen test seeds / terrains 별도 보유.
- 최소 3-5개의 학습 seed를 같은 조건에서 짝지어 사용. 극단 성능을 보려면 seed 간 worst 또는 CVaR 제약을 추가.
- `train_seeds`를 고정하면 서로 다른 후보 간 stochastic training seed 변동을 제어함.
- `research_git_revision`과 run별 `parameters.json`, checkpoint 저장. 실제 MDP config, simulator / GPU / randomization 버전도 별도 로그에 유지.

### 원문 API 참고

- Isaac Lab 2.3 Hydra: https://isaac-sim.github.io/IsaacLab/v2.3.2/source/features/hydra.html
- Isaac Lab 2.3 RL scripts: https://docs.nvidia.com/isaaclab/2.3/overview/reinforcement-learning-scripts
- RSL-RL resume cfg: https://isaac-sim.github.io/IsaacLab/v2.3.1/source/api/lab_rl/isaaclab_rl.html

import json
import asyncio
from backend.env.sumo_env import SumoTrafficEnv
from backend.ws_state import ws_clients   # 필요 없음 (삭제 가능)

SAVE_DIR = "frontend/static"   # 프론트에서 접근 가능한 폴더

# -------------------------------
# 1) 한 에피소드 실행 + 프레임 기록
# -------------------------------
import asyncio
from backend.env.sumo_env import SumoTrafficEnv
from backend.replay_store import replay_store

async def run_episode_and_record(env, policy_fn, model_name, max_steps=1000):
    obs, info = env.reset()
    done = False
    truncated = False

    rewards = []
    steps = 0

    frames = []   # 메모리에 저장할 프레임 리스트

    while not (done or truncated) and steps < max_steps:

        # baseline action = 0
        if policy_fn is None:
            action = 0
        else:
            action = policy_fn(obs)

        obs, reward, done, truncated, info = env.step(action)

        frame_state = env.get_vis_state()
        frame_state["step"] = steps
        frames.append(frame_state)

        rewards.append(float(reward))
        steps += 1

        await asyncio.sleep(0)

    # 🔥 JSON 파일 저장 제거
    # 저장소에 메모리로 저장
    replay_store[model_name] = {
        "frames": frames,
        "rewards": rewards,
        "total_reward": sum(rewards),
    }

    print(f"[STORE] Saved {model_name} replay in memory ({len(frames)} frames)")

    return replay_store[model_name]




# -------------------------------
# 2) baseline / DQN / PPO 실행
# -------------------------------
async def evaluate_scenario(dqn_agent, ppo_agent, scenario_config):

    results = {}

    # Baseline
    env = SumoTrafficEnv(scenario_config)
    results["baseline"] = await run_episode_and_record(
        env, None, model_name="baseline"
    )
    env.close()

    # DQN
    env = SumoTrafficEnv(scenario_config)
    results["dqn"] = await run_episode_and_record(
        env, lambda obs: dqn_agent.select_action(obs, epsilon=0.0),
        model_name="dqn"
    )
    env.close()

    # PPO
    env = SumoTrafficEnv(scenario_config)
    results["ppo"] = await run_episode_and_record(
        env, ppo_action_wrapper(ppo_agent),
        model_name="ppo"
    )
    env.close()

    return results



# -------------------------------
# PPO wrapper 그대로 유지
# -------------------------------
def ppo_action_wrapper(ppo_agent):
    def wrapper(obs):
        out = ppo_agent.select_action(obs)

        if isinstance(out, tuple):
            action = out[0]
        else:
            action = out

        if hasattr(action, "item"):
            action = action.item()

        if isinstance(action, (list, tuple)):
            action = action[0]

        return int(action)
    return wrapper

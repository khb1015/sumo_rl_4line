from fastapi import FastAPI, WebSocket, APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import torch
import os
import asyncio
import json

from backend.env.scenario import build_scenario
from backend.services.evaluate import evaluate_scenario
from backend.models.dqn_agent import DQNAgent
from backend.models.ppo_agent import PPOAgent
from backend.ws_state import ws_clients
from fastapi import APIRouter
from backend.replay_store import replay_store

app = FastAPI()

router = APIRouter()

# ================================
# 1) Frontend static 파일 서빙
# ================================
FRONTEND_DIR = "frontend"
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root():
    """ index.html 반환 """
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


router = APIRouter()

@router.get("/api/replay")
def get_replay(model: str):
    if model not in replay_store or replay_store[model] is None:
        return {"error": "no replay data"}

    return replay_store[model]   # JSON 파일로 저장한 적 없어도 바로 반환됨

# 라우터 등록
app.include_router(router)


# ================================
# 2) 모델 로드
# ================================
DQN_PATH = "pretrained/dqn_torch_final.pth"
PPO_PATH = "pretrained/ppo_model"

dqn_agent = DQNAgent(6, 2)
dqn_agent.policy_net.load_state_dict(torch.load(DQN_PATH, map_location="cpu"))

ppo_agent = PPOAgent(state_dim=6, action_dim=2, rollout_steps=10000)
ppo_data = torch.load(PPO_PATH, map_location="cpu")
ppo_agent.actor.load_state_dict(ppo_data["actor"])
ppo_agent.critic.load_state_dict(ppo_data["critic"])
ppo_agent.optimizer.load_state_dict(ppo_data["optimizer"])


# ================================
# 3) 요청 스키마
# ================================
class ScenarioRequest(BaseModel):
    scenario: str                          # "A", "B", "C", "D", "custom"
    custom_flows: dict | None = None       # custom이면 채워짐



# ================================
# 4) 평가 API
# ================================
@app.post("/api/evaluate")
async def evaluate(req: ScenarioRequest):
    scenario_config = build_scenario(
        scenario_name=req.scenario,
        custom_flows=req.custom_flows
    )



    results = await evaluate_scenario(
        dqn_agent,
        ppo_agent,
        scenario_config=scenario_config
    )
    return results


@app.websocket("/ws/state")
async def ws_state(websocket: WebSocket):
    await websocket.accept()
    ws_clients.add(websocket)

    try:
        while True:
            await asyncio.sleep(1)
    except:
        ws_clients.remove(websocket)

# ================================
# 5) 상태 확인
# ================================
@app.get("/api/status")
def status():
    return {"message": "RL Traffic Web API is running"}

**SUMO 기반 4차선 교차로 강화학습 신호 제어 + 웹 시각화 시스템**

# 주요 기능

### SUMO를 통한 실제 도로 기반 시뮬레이션  
- 4차선 교차로 모델링  
- 차량 스폰/제거 로직  
- 시간대별 시나리오 기반 traffic 패턴 적용

### 강화학습 기반 신호 제어  
- **DQN**(Deep Q-Network)  
- **PPO**(Proximal Policy Optimization)  
- Baseline(전통적 Fixed-Time)

### 웹 시각화 대시보드 제공  
- 총 보상 비교  
- 보상 그래프 차트  
- SUMO 2D replay (차량·신호등 상태 시각화)

### Replay 저장 방식  
- JSON 파일이 아니라 **메모리 저장 (replay_store)** → 캐싱 문제 없음  
- Web에서 `/api/replay?model=dqn` 방식으로 재생 가능

# 
프로젝트 구조
web/
├─ backend/
│  ├─ app.py              # FastAPI 서버 (API 엔트리)
│  ├─ ws_state.py            
│  ├─ replay_store.py           
│  ├─ env/
│  │  ├─ sumo_env.py      # 이미 쓰고 있는 SumoTrafficEnv (평가용)
│  │  └─ scenario.py      # time_periods/traffic 설정 로직
│  ├─ models/
│  │  ├─ dqn_agent.py     # 학습된 DQN 래퍼 (load + select_action)
│  │  └─ ppo_agent.py     # 학습된 PPO 래퍼
│  └─ services/
│     └─ evaluate.py      # baseline / DQN / PPO를 같은 시나리오에서 평가
├─ frontend/
│  ├─ index.html          # UI + 차트
│  ├─ static/
│  │   └─js/
│  │   │  ├─ api.js           # 백엔드 호출
│  │   │  └─ chart.js         # Chart.js로 그래프 그리기
│  │   └─ css/
│  │   │  └─ style.css
├─ sumo/
│  ├─ four_lane.net.xml   # 4차선 도로 네트워크
│  └─ (sumocfg 등 필요하면)
└─ pretrained/
├─ dqn_four_lane.pt    # 학습 완료된 DQN 가중치
└─ ppo_four_lane.pt    # 학습 완료된 PPO 가중치

#패키지 목록
fastapi
uvicorn
python-multipart
pydantic
numpy
gymnasium
torch
matplotlib

#실행
uvicorn backend.app:app --reload --port 8000를 통해 실행 가능


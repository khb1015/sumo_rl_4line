
# ===============================================================
# ================== 4. 강화학습 SUMO 환경 ======================
# ===============================================================
import numpy as np
import traci, random
import gymnasium as gym
from gymnasium import spaces
SUMO_CFG = r"sumo\sejong.sumocfg"
TLS_ID_AUTO = True          # True면 첫 번째 신호등 자동 선택
TLS_ID_FALLBACK = "tls_0" 
SPAWN_INTERVAL = 10      # 몇 step마다 스폰 시도할지
MAX_STEPS = 1000    
SUMO_BIN = "sumo"

EDGES_DATA = [
    ("976906315#0",      8952652972, 9651134449),
    ("976906314#1",      5120772423, 1066203094),
    ("-416819990#0",     3991887966, 4175105101),
    ("416819990#0",      4175105101, 3991887966),
    ("976906314#3",      9651134455, 436832680),
    ("976906312#0",      8952652973, 9651134456),
    ("175320504#1",      7856245557, 4177516258),
    ("-175320504#1",     4177516258, 7856245557),
    ("-175320504#0",     7856245557, 8952652972),
    ("175320504#0",      8952652972, 7856245557),
    ("-976906316",       8952652972, 1066203094),
    ("976906316",        1066203094, 8952652972),
    ("-416819990#1",     1066203094, 3991887966),
    ("416819990#1",      3991887966, 1066203094),
    ("976906314#2",      1066203094, 9651134455),
    ("976906312#1",      9651134456, 8952652972),
]


SCENARIOS = {
    "scenario_A": {  # 보통 패턴 (아침 ↑ 점심 ↓ 저녁 ↑)
        (0, 300): {
            "4175105101": (0.03, 200),
            "8952652973": (0.03, 200),
            "4177516258": (0.03, 200),
            "5120772423": (0.03, 200),
        },
        (300, 600): {
            "4175105101": (0.01, 180),
            "8952652973": (0.01, 180),
            "4177516258": (0.01, 180),
            "5120772423": (0.01, 180),
        },
        (600, 900): {
            "4175105101": (0.03, 220),
            "8952652973": (0.03, 220),
            "4177516258": (0.03, 220),
            "5120772423": (0.03, 220),
        }
    },

    "scenario_B": {  # 점심이 제일 혼잡한 패턴
        (0, 300): {
            "4175105101": (0.01, 180),
            "8952652973": (0.01, 180),
            "4177516258": (0.01, 180),
            "5120772423": (0.01, 180),
        },
        (300, 600): {
            "4175105101": (0.04, 230),
            "8952652973": (0.04, 230),
            "4177516258": (0.04, 230),
            "5120772423": (0.04, 230),
        },
        (600, 900): {
            "4175105101": (0.02, 200),
            "8952652973": (0.02, 200),
            "4177516258": (0.02, 200),
            "5120772423": (0.02, 200),
        }
    },

    "scenario_C": {  # 하루종일 낮은 트래픽
        (0, 300): {
            "4175105101": (0.01, 150),
            "8952652973": (0.01, 150),
            "4177516258": (0.01, 150),
            "5120772423": (0.01, 150),
        },
        (300, 600): {
            "4175105101": (0.01, 150),
            "8952652973": (0.01, 150),
            "4177516258": (0.01, 150),
            "5120772423": (0.01, 150),
        },
        (600, 900): {
            "4175105101": (0.01, 150),
            "8952652973": (0.01, 150),
            "4177516258": (0.01, 150),
            "5120772423": (0.01, 150),
        }
    },

    "scenario_D": {  # 하루종일 매우 높은 트래픽
        (0, 300): {
            "4175105101": (0.04, 250),
            "8952652973": (0.04, 250),
            "4177516258": (0.04, 250),
            "5120772423": (0.04, 250),
        },
        (300, 600): {
            "4175105101": (0.04, 250),
            "8952652973": (0.04, 250),
            "4177516258": (0.04, 250),
            "5120772423": (0.04, 250),
        },
        (600, 900): {
            "4175105101": (0.04, 250),
            "8952652973": (0.04, 250),
            "4177516258": (0.04, 250),
            "5120772423": (0.04, 250),
        }
    }
}

# ================== 2. 노드 기반 spawn / despawn ==================
SPAWNABLE_NODES = [
    "4175105101",
    "8952652973",
    "4177516258",
    "5120772423"
]

DESPAWNABLE_NODES = [
    "4175105101",
    "4177516258",
    "436832680",
    "9651134449"
]

node_to_out_edges = {}
node_to_in_edges = {}

for eid, frm, to in EDGES_DATA:
    frm, to = str(frm), str(to)
    node_to_out_edges.setdefault(frm, []).append(eid)
    node_to_in_edges.setdefault(to, []).append(eid)


def pick_start_edge(start_node: str):
    edges = node_to_out_edges.get(str(start_node), [])
    if not edges:
        return None
    return random.choice(edges)


def pick_end_edge(end_node: str):
    edges = node_to_in_edges.get(str(end_node), [])
    if not edges:
        return None
    return random.choice(edges)



class SumoTrafficEnv(gym.Env):
    """
    - 액션: 0=신호 유지, 1=다음 페이즈로 전환
    - 관측: [total_waiting_time, mean_speed, queue_len] (tanh 정규화)
    - 보상: -(α*대기시간 + β*대기행렬) + γ*평균속도 - delta*(phase 변경 여부)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, scenario_config=None, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.sumo_started = False
        self.sim_step = 0
        self.last_action = 0
        self.delta = 1 # phase 변경 penalty
        if scenario_config is not None:
            self.current_scenario_name = scenario_config.get("scenario", "custom")
            self.current_scenario = scenario_config["time_periods"]
        else:
            self.current_scenario_name = "scenario_C"
            self.current_scenario = SCENARIOS["scenario_C"]


        # 액션: 유지 / 다음 페이즈
        self.action_space = spaces.Discrete(2)

        # 관측: 3차원 실수
        obs_low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # 보상 가중치
        self.alpha = 0.002  # waiting_time penalty
        self.beta = 0.05    # queue length penalty
        self.gamma = 0.01   # mean speed reward

        # TLS 관련
        self.tls_id = None
        self.n_phases = 1

        if scenario_config is not None:
            self.current_scenario = scenario_config["time_periods"]
        else:
            self.current_scenario = SCENARIOS["scenario_C"] 

        # ---------- SUMO 관리 ----------
    def _start_sumo(self):
        if self.sumo_started:
            return

        # render_mode == "human"이면 sumo-gui 사용
        sumo_bin = "sumo-gui" if self.render_mode == "human" else SUMO_BIN

        traci.start([sumo_bin, "-c", SUMO_CFG, "--start", "--no-step-log"
                     , "--seed", "42", "--random", "false"])
        self.sumo_started = True

        tls_list = traci.trafficlight.getIDList()
        if TLS_ID_AUTO and len(tls_list) > 0:
            self.tls_id = tls_list[0]
        else:
            self.tls_id = TLS_ID_FALLBACK if TLS_ID_FALLBACK in tls_list else (tls_list[0] if tls_list else None)

        if self.tls_id is not None:
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
            self.n_phases = len(logic.getPhases())
        else:
            self.n_phases = 1  # 신호 없으면 의미상 1

    def _close_sumo(self):
        if self.sumo_started:
            traci.close()
            self.sumo_started = False


    
    # ---------- Gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # SUMO 종료 후 재시작
        try:
            traci.close()
        except:
            pass

        self.sumo_started = False

        # 시나리오는 __init__에서 이미 설정됨 → reset에서 변경 금지
        print(f"[ENV] Selected Scenario: {self.current_scenario_name}")

        self._start_sumo()
        self.sim_step = 0
        self.last_action = 0

        print("[ENV] Reset done, vehicle count =", len(traci.vehicle.getIDList()))

        return self._get_obs(), {}


    def _get_direction(self, veh_id):
        parts = veh_id.split("_")
        if len(parts) < 3:
            return None
        
        node = parts[1]
        if node == "4175105101": 
            return "E"   # 동
        if node == "4177516258":
            return "W"   # 서
        if node == "8952652973":
            return "S"   # 남
        if node == "5120772423":
            return "N"   # 북
        return None

        

    def step(self, action):
        self.last_action = int(action)

        # 1) TLS 액션 적용
        if self.tls_id is not None and self.n_phases > 1:
            if self.last_action == 1:
                cur = traci.trafficlight.getPhase(self.tls_id)
                traci.trafficlight.setPhase(self.tls_id, (cur + 1) % self.n_phases)
            # 0이면 유지

        # 2) 스폰 / 제거
        self._spawn_and_remove()

        # 3) 시뮬레이션 1 step
        traci.simulationStep()
        self.sim_step += 1

        # 4) 관측/보상/종료
        obs = self._get_obs()
        reward = self._get_reward()

        terminated = False
        truncated = (self.sim_step >= MAX_STEPS)
        info = {}

        return obs, reward, terminated, truncated, info
    
    def _active_period(self):
        for (start, end), pattern in self.current_scenario.items():
            if start <= self.sim_step < end:
                return pattern
        return None
    def set_scenario(self, scenario_config):
        # 사용자가 직접 만든 dict → 바로 적용
        if isinstance(scenario_config, dict):
            self.current_scenario = scenario_config["time_periods"]
            self.current_scenario_name = "custom"
            return

        # 문자열 기반 SCENARIOS 사용
        if scenario_config not in SCENARIOS:
            raise ValueError(f"Scenario {scenario_config} not found.")

        self.current_scenario_name = scenario_config
        self.current_scenario = SCENARIOS[scenario_config]





    # ---------- 시나리오 로직 ----------
    def _spawn_and_remove(self):
        # 현재 scenario에서 시간대별 패턴 가져오기
        pattern = self._active_period()
        if pattern is None:
            return

        # ---- 스폰 ----
        if SPAWN_INTERVAL > 0 and self.sim_step % SPAWN_INTERVAL == 0:

            for spawn_node, (spawn_rate, _) in pattern.items():
                if random.random() < spawn_rate:
                    end_node = random.choice(DESPAWNABLE_NODES)

                    start_edge = pick_start_edge(spawn_node)
                    end_edge = pick_end_edge(end_node)

                    if not start_edge or not end_edge:
                        continue

                    route_id = f"r_{spawn_node}_{end_node}_{self.sim_step}"
                    veh_id = f"veh_{spawn_node}_{self.sim_step}"

                    try:
                        traci.route.add(route_id, [start_edge, end_edge])
                        traci.vehicle.add(veh_id, route_id, typeID="car")
                    except traci.TraCIException:
                        pass

        # ---- 제거 ----
        veh_ids = traci.vehicle.getIDList()
        for vid in veh_ids:

            if not vid.startswith("veh_"):
                continue

            try:
                depart_t = traci.vehicle.getDeparture(vid)
            except traci.TraCIException:
                continue

            parts = vid.split("_")
            if len(parts) < 3:
                continue

            spawn_node = parts[1]

            # scenario에서 removal_delay 찾기
            removal_delay = None
            for (_, p) in self.current_scenario.items():
                if spawn_node in p:
                    removal_delay = p[spawn_node][1]
                    break

            if removal_delay is None:
                removal_delay = 200

            if self.sim_step - depart_t > removal_delay:
                try:
                    traci.vehicle.remove(vid)
                except traci.TraCIException:
                    pass

    # ---------- 관측/보상 ----------
    def _get_obs(self):
        try:
            veh_ids = traci.vehicle.getIDList()
            total_wait = sum(traci.vehicle.getWaitingTime(v) for v in veh_ids)
            speeds = [traci.vehicle.getSpeed(v) for v in veh_ids] if veh_ids else [0.0]
            mean_speed = float(np.mean(speeds))
        except traci.TraCIException:
            total_wait, mean_speed = 0.0, 0.0

        # ---------- 방향별 queue 계산 ----------
        q_E = q_W = q_S = q_N = 0
        
        for v in veh_ids:
            try:
                if traci.vehicle.getSpeed(v) < 0.1:
                    d = self._get_direction(v)
                    if d == "E": q_E += 1
                    elif d == "W": q_W += 1
                    elif d == "S": q_S += 1
                    elif d == "N": q_N += 1
            except:
                pass

        # ---------- 정규화 ----------
        obs = np.array([
            q_E / 20.0,
            q_W / 20.0,
            q_S / 20.0,
            q_N / 20.0,
            np.tanh(total_wait / 500.0),
            np.tanh(mean_speed / 15.0),
        ], dtype=np.float32)

        return obs


    def _get_reward(self):
        try:
            veh_ids = traci.vehicle.getIDList()
            total_wait = sum(traci.vehicle.getWaitingTime(v) for v in veh_ids)
            queue_len = sum(1 for v in veh_ids if traci.vehicle.getSpeed(v) < 0.1)
            speeds = [traci.vehicle.getSpeed(v) for v in veh_ids] if veh_ids else [0.0]
            mean_speed = float(np.mean(speeds))
        except traci.TraCIException:
            total_wait, queue_len, mean_speed = 0.0, 0, 0.0

        reward = - self.alpha * total_wait - self.beta * queue_len + self.gamma * mean_speed

        # 신호 변경 패널티
        if self.last_action == 1:
            reward -= self.delta

        return float(reward)
    
    def get_vis_state(self):
        vehicles = []
        for v in traci.vehicle.getIDList():
            x, y = traci.vehicle.getPosition(v)
            angle = traci.vehicle.getAngle(v)
            vehicles.append({
                "id": v,
                "x": x,
                "y": y,
                "angle": angle
            })
        tls = traci.trafficlight.getPhase(self.tls_id)
        return {"vehicles": vehicles, "tls": tls}

    def close(self):
        try:
            if self.sumo_started:
                traci.close(False)
        except Exception:
            pass  # 이미 종료된 경우

        self.sumo_started = False
        return super().close()
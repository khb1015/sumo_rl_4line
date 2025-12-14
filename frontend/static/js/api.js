// frontend/static/js/api.js

// 1) 시뮬레이션 실행 함수 (반드시 async)
// frontend/static/js/api.js

// 1) 시뮬레이션 실행 함수
async function runSimulation(payload) {
    console.log("=== Sending request JSON ===");
    console.log(JSON.stringify(payload, null, 2));

    try {
        const response = await fetch("/api/evaluate", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            console.error("Server error:", response.status);
            return;
        }

        const result = await response.json();
        console.log("Result:", result);

        document.getElementById("baseline_reward").innerText =
            result.baseline.total_reward.toFixed(2);
        document.getElementById("dqn_reward").innerText =
            result.dqn.total_reward.toFixed(2);
        document.getElementById("ppo_reward").innerText =
            result.ppo.total_reward.toFixed(2);

        drawRewardChart(result);

    } catch (err) {
        console.error("Fetch error:", err);
    }
}


// 2) 버튼 이벤트 (payload 생성은 여기서만!)
document.addEventListener("DOMContentLoaded", () => {
    const runBtn = document.getElementById("runBtn");

    runBtn.onclick = async () => {

        const scenarioName = document.getElementById("scenario_select")?.value;

        if (!scenarioName) {
            console.error("scenario_select 값을 읽을 수 없습니다.");
            return;
        }

        // 기본 payload
        let payload = { scenario: scenarioName };

        // Custom Scenario면 custom_flows 추가
        if (scenarioName === "custom") {
            payload.custom_flows = {};

            document.querySelectorAll(".custom-rate").forEach(input => {
                const period = input.dataset.period;
                const node = input.dataset.node;
                const value = parseFloat(input.value);

                if (!payload.custom_flows[period])
                    payload.custom_flows[period] = {};

                payload.custom_flows[period][node] = value;
            });
        }

        // 🔥 이제 반드시 runSimulation(payload) 호출
        runSimulation(payload);
    };
});



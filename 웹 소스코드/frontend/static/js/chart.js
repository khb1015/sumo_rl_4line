let rewardChart = null;

function drawRewardChart(data) {
    const ctx = document.getElementById("rewardChart").getContext("2d");

    const steps = data.dqn.rewards.map((_, i) => i + 1);

    if (rewardChart) rewardChart.destroy();

    rewardChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: steps,
            datasets: [
                {
                    label: "DQN Reward",
                    data: data.dqn.rewards,
                    borderColor: "blue",
                    borderWidth: 2,
                    tension: 0.2
                },
                {
                    label: "PPO Reward",
                    data: data.ppo.rewards,
                    borderColor: "green",
                    borderWidth: 2,
                    tension: 0.2
                },
                {
                    label: "Baseline Reward",
                    data: data.baseline.rewards,
                    borderColor: "red",
                    borderDash: [5, 5],
                    borderWidth: 2,
                    tension: 0.2
                }
            ]
        }
    });
}

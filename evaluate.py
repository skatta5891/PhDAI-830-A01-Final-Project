import csv
import json
from pathlib import Path

import numpy as np
import torch

from baselines import fixed_action, threshold_action
from dqn import QNetwork, greedy_action
from environment import ServiceEnvironment
from workloads import load_config, make_traces


ROOT = Path(__file__).parent
OUTPUT = ROOT / "outputs"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rollout(config, trace, policy):
    env = ServiceEnvironment(config)
    observation, _ = env.reset(seed=trace["seed"], options={"trace": trace})
    logs = []
    while True:
        action = policy(env, observation)
        observation, _, terminal, _, info = env.step(action)
        logs.append(info)
        if terminal:
            break
    metrics = {
        "trace_seed": trace["seed"], "cost": sum(row["billed"] for row in logs),
        "arrivals": sum(row["arrivals"] for row in logs),
        "unserved": sum(row["dropped"] + row["terminal_queue"] for row in logs),
        "mean_queue": float(np.mean([row["queue"] for row in logs])),
        "queue_above_20_pct": 100 * np.mean([row["queue"] > 20 for row in logs]),
        "scaling_actions": sum(row["changed"] for row in logs),
        "return": sum(row["reward"] for row in logs),
    }
    return metrics, logs


def aggregate(rows):
    # Use total arrivals for the unserved fraction, not a mean of episode ratios.
    return {
        "cost": float(np.mean([row["cost"] for row in rows])),
        "unserved_fraction": sum(row["unserved"] for row in rows) / sum(row["arrivals"] for row in rows),
        **{key: float(np.mean([row[key] for row in rows])) for key in
           ["mean_queue", "queue_above_20_pct", "scaling_actions", "return"]},
    }


def validation_return(config, traces, policy):
    return float(np.mean([rollout(config, trace, policy)[0]["return"] for trace in traces]))


def load_network(path, config):
    network = QNetwork(config)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    network.load_state_dict(checkpoint["model"])
    network.eval()
    return network


def main():
    torch.set_num_threads(1)
    config = load_config()
    traces = make_traces(config)
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "traces.json").write_text(json.dumps(traces))
    candidates = []
    for target in range(1, config["max_replicas"] + 1):
        score = validation_return(config, traces["validation"], lambda env, obs: fixed_action(env, target))
        candidates.append({"controller": "fixed", "target": target, "upper": "", "lower": "", "validation_return": score})
    for upper in config["upper_thresholds"]:
        for lower in config["lower_thresholds"]:
            score = validation_return(config, traces["validation"], lambda env, obs: threshold_action(env, upper, lower))
            candidates.append({"controller": "threshold", "target": "", "upper": upper, "lower": lower, "validation_return": score})
    write_csv(OUTPUT / "baseline_validation.csv", candidates)
    fixed = max([row for row in candidates if row["controller"] == "fixed"], key=lambda row: row["validation_return"])
    threshold = max([row for row in candidates if row["controller"] == "threshold"], key=lambda row: row["validation_return"])
    selections = {"fixed": fixed, "threshold": threshold, "dqn": []}
    for seed in config["training_seeds"]:
        history = json.loads((OUTPUT / f"run_{seed}" / "validation.json").read_text())
        best = max(history, key=lambda row: row["validation_return"])
        selections["dqn"].append({"seed": seed, **best})
    # Freeze all selections before held-out evaluation.
    (OUTPUT / "selections.json").write_text(json.dumps(selections, indent=2))
    policies = [("Fixed", 0, lambda env, obs: fixed_action(env, fixed["target"])),
                ("Threshold", 0, lambda env, obs: threshold_action(env, threshold["upper"], threshold["lower"]))]
    for selection in selections["dqn"]:
        seed, steps = selection["seed"], selection["steps"]
        network = load_network(OUTPUT / f"run_{seed}" / f"checkpoint_{steps}.pt", config)
        # Bind this network now so each policy retains its own training run.
        policies.append(("DQN", seed, lambda env, obs, net=network: greedy_action(net, obs)))
    summaries, episode_rows = [], []
    for workload in ["nominal", "robustness"]:
        for name, seed, policy in policies:
            metrics = []
            for index, trace in enumerate(traces[workload + "_test"]):
                row, logs = rollout(config, trace, policy)
                row.update({"workload": workload, "controller": name, "training_seed": seed})
                metrics.append(row)
                episode_rows.append(row)
                if index == 0:
                    # Demonstrate the first trace by a fixed rule, without cherry-picking.
                    write_csv(OUTPUT / "replay" / f"{workload}_{name}_{seed}.csv", logs)
            summaries.append({"workload": workload, "controller": name, "training_seed": seed, **aggregate(metrics)})
    write_csv(OUTPUT / "episodes.csv", episode_rows)
    write_csv(OUTPUT / "summary.csv", summaries)
    assessment = {}
    metric_names = ("cost", "unserved_fraction", "mean_queue",
                    "queue_above_20_pct", "scaling_actions", "return")
    for workload in ["nominal", "robustness"]:
        baseline = next(row for row in summaries if row["workload"] == workload and row["controller"] == "Threshold")
        runs = [row for row in summaries if row["workload"] == workload and row["controller"] == "DQN"]
        # Summarize learning variation across all three runs, including failures.
        means = {key: float(np.mean([row[key] for row in runs])) for key in metric_names}
        deviations = {key: float(np.std([row[key] for row in runs], ddof=1)) for key in means}
        saving = 100 * (baseline["cost"] - means["cost"]) / baseline["cost"]
        conditions = {"saving_at_least_10_pct": saving >= 10,
                      "no_unserved_increase": means["unserved_fraction"] <= baseline["unserved_fraction"],
                      "queue_increase_at_most_1_pp": means["queue_above_20_pct"] <= baseline["queue_above_20_pct"] + 1}
        assessment[workload] = {"dqn_mean": means, "dqn_sample_sd": deviations,
                                "threshold": baseline, "saving_pct": saving,
                                "conditions": conditions, "all_conditions_met": all(conditions.values())}
    (OUTPUT / "assessment.json").write_text(json.dumps(assessment, indent=2))
    print(json.dumps(assessment, indent=2), flush=True)


if __name__ == "__main__":
    main()

import argparse
import copy
import json
import platform
import time

import gymnasium
import numpy as np
import torch

from dqn import QNetwork, ReplayBuffer, greedy_action, update_network
from environment import ServiceEnvironment
from evaluate import OUTPUT, validation_return, write_csv
from workloads import load_config, make_traces


def train(seed, config):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # Keep exploration and replay sampling on separate random generators.
    replay_rng = np.random.default_rng(seed + 1000)
    network = QNetwork(config)
    target = copy.deepcopy(network)
    optimizer = torch.optim.Adam(network.parameters(), lr=config["learning_rate"])
    replay = ReplayBuffer(config["replay_capacity"])
    env = ServiceEnvironment(config)
    observation, _ = env.reset(seed=seed)
    training_trace_seeds = [env.trace["seed"]]
    validation = make_traces(config)["validation"]
    directory = OUTPUT / f"run_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    episodes, losses, history = [], [], []
    episode_return, episode_steps = 0.0, 0
    start = time.perf_counter()
    for step in range(1, config["training_steps"] + 1):
        # Epsilon is the chance of choosing a random action.
        fraction = min((step - 1) / config["epsilon_decay_steps"], 1.0)
        epsilon = config["epsilon_start"] + fraction * (config["epsilon_end"] - config["epsilon_start"])
        action = int(rng.integers(3)) if rng.random() < epsilon else greedy_action(network, observation)
        next_observation, reward, terminal, _, _ = env.step(action)
        replay.add(observation, action, reward, next_observation, terminal)
        episode_return += reward
        episode_steps += 1
        if step >= config["learning_starts"] and step % config["train_every"] == 0:
            loss = update_network(network, target, optimizer, replay.sample(replay_rng, config["batch_size"]), config)
            losses.append({"steps": step, "loss": loss, "epsilon": epsilon})
        if step % config["target_every"] == 0:
            # Hold target weights fixed between scheduled copies.
            target.load_state_dict(network.state_dict())
        observation = next_observation
        if terminal:
            # Only completed episodes enter the episode return table.
            episodes.append({"steps": step, "episode": len(episodes) + 1,
                             "return": episode_return, "length": episode_steps, "epsilon": epsilon})
            observation, _ = env.reset()
            training_trace_seeds.append(env.trace["seed"])
            episode_return, episode_steps = 0.0, 0
        if step % config["checkpoint_every"] == 0:
            # Save and score checkpoints on validation traffic, without test data.
            torch.save({"model": network.state_dict(), "steps": step, "seed": seed, "config": config},
                       directory / f"checkpoint_{step}.pt")
            score = validation_return(config, validation, lambda service, obs: greedy_action(network, obs))
            history.append({"steps": step, "validation_return": score})
            (directory / "validation.json").write_text(json.dumps(history, indent=2))
            write_csv(directory / "episodes.csv", episodes)
            write_csv(directory / "losses.csv", losses)
            print(f"seed={seed} steps={step} validation_return={score:.3f}", flush=True)
    metadata = {"seed": seed, "training_steps": config["training_steps"],
                "elapsed_seconds": time.perf_counter() - start,
                "completed_episodes": len(episodes), "unfinished_episode_steps": episode_steps,
                "python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__, "gymnasium": gymnasium.__version__,
                "device": "cpu", "torch_threads": 1, "config": config,
                "training_trace_seeds": training_trace_seeds}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Finished seed {seed} in {metadata['elapsed_seconds']:.1f} seconds", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_config()
    for seed in [args.seed] if args.seed is not None else config["training_seeds"]:
        train(seed, config)

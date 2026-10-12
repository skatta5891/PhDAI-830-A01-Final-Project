import gymnasium as gym
import numpy as np
from gymnasium import spaces

from workloads import generate_trace, load_config


class ServiceEnvironment(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None):
        self.config = config or load_config()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.array([np.inf, 1, 1, 1, 1, 1], dtype=np.float32),
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        trace_seed = int(self.np_random.integers(0, 2**31))
        self.trace = options.get("trace")
        if self.trace is None:
            self.trace = generate_trace(trace_seed, self.config)
        if len(self.trace["arrivals"]) != self.config["episode_intervals"]:
            raise ValueError("Trace length must equal episode_intervals")
        self.interval = 0
        self.active = self.config["initial_replicas"]
        self.timer = 0
        self.queue = 0
        return self.observation(), {}

    def observation(self):
        # A terminal observation has no next arrival to process.
        if self.interval == self.config["episode_intervals"]:
            arrivals, mean = 0, 0
        else:
            arrivals = self.trace["arrivals"][self.interval]
            mean = self.trace["means"][self.interval]
        return np.array([
            arrivals / 100.0, mean / 60.0,
            self.active / self.config["max_replicas"],
            self.timer / self.config["startup_intervals"],
            self.queue / self.config["queue_capacity"],
            self.interval / self.config["episode_intervals"],
        ], dtype=np.float32)

    def step(self, action):
        if self.interval >= self.config["episode_intervals"]:
            raise RuntimeError("Reset the environment after termination")
        if not self.action_space.contains(action):
            raise ValueError("Action must be 0, 1, or 2")
        changed = 0
        # Apply scaling before calculating service capacity and interval cost.
        if action == 0 and self.active > 1:
            self.active -= 1
            changed = 1
        elif action == 2 and self.timer == 0 and self.active < self.config["max_replicas"]:
            self.timer = self.config["startup_intervals"]
            changed = 1
        billed = self.active + int(self.timer > 0)
        # Starting replicas incur cost but cannot serve requests yet.
        arrivals = int(self.trace["arrivals"][self.interval])
        previous_queue = self.queue
        capacity = self.active * self.config["capacity_per_replica"]
        served = min(previous_queue + arrivals, capacity)
        waiting = previous_queue + arrivals - served
        self.queue = min(waiting, self.config["queue_capacity"])
        dropped = waiting - self.queue
        # Count the final queue as unserved so the policy cannot defer work forever.
        terminal = self.interval + 1 == self.config["episode_intervals"]
        terminal_queue = self.queue if terminal else 0
        weights = self.config["reward_weights"]
        reward = -(weights[0] * billed + weights[1] * self.queue
                   + weights[2] * (dropped + terminal_queue) + weights[3] * changed)
        info = {
            "interval": self.interval, "arrivals": arrivals,
            "workload_mean": int(self.trace["means"][self.interval]),
            "previous_queue": previous_queue, "active": self.active,
            "pending": int(self.timer > 0), "startup_timer": self.timer,
            "capacity": capacity, "served": served, "queue": self.queue,
            "dropped": dropped, "terminal_queue": terminal_queue,
            "billed": billed, "changed": changed, "action": int(action),
            "reward": reward,
        }
        if self.timer > 0:
            # Advance startup after service. An addition in t first serves in t + 2.
            self.timer -= 1
            if self.timer == 0:
                self.active += 1
        self.interval += 1
        return self.observation(), float(reward), terminal, False, info

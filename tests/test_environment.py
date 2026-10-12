import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env

from baselines import fixed_action, threshold_action
from dqn import QNetwork, ReplayBuffer, update_network
from environment import ServiceEnvironment
from evaluate import aggregate, rollout
from workloads import generate_trace, load_config, make_traces


def setup_env(arrivals, **changes):
    config = load_config()
    config.update(changes)
    config["episode_intervals"] = len(arrivals)
    env = ServiceEnvironment(config)
    trace = {"seed": 1, "arrivals": arrivals, "means": [20] * len(arrivals)}
    env.reset(seed=1, options={"trace": trace})
    return env


def test_two_complete_startup_intervals_and_billing():
    env = setup_env([40, 40, 40])
    first = env.step(2)[4]
    second = env.step(1)[4]
    third = env.step(1)[4]
    assert [first["active"], second["active"], third["active"]] == [3, 3, 4]
    assert [first["billed"], second["billed"], third["billed"]] == [4, 4, 4]
    assert [first["served"], second["served"], third["served"]] == [30, 30, 40]


def test_replica_bounds_and_pending_limit():
    env = setup_env([0] * 30)
    for _ in range(10):
        row = env.step(0)[4]
        assert row["active"] >= 1
    for _ in range(20):
        row = env.step(2)[4]
        assert row["billed"] <= 10
    assert env.active == 10


def test_invalid_add_and_remove_have_no_change_penalty():
    env = setup_env([0] * 4, initial_replicas=1)
    row = env.step(0)[4]
    assert row["changed"] == 0 and row["billed"] == 1
    env.step(2)
    row = env.step(2)[4]
    assert row["changed"] == 0 and row["pending"] == 1


def test_removal_is_immediate():
    env = setup_env([30])
    row = env.step(0)[4]
    assert row["served"] == 20 and row["billed"] == 2


def test_overflow_and_terminal_queue_penalty():
    env = setup_env([300], initial_replicas=1)
    observation, reward, terminal, truncated, row = env.step(1)
    assert row["served"] == 10 and row["queue"] == 200 and row["dropped"] == 90
    assert row["terminal_queue"] == 200
    assert reward == pytest.approx(-(0.15 + 0.04 * 200 + 0.25 * 290))
    assert terminal and not truncated and observation[-1] == 1
    with pytest.raises(RuntimeError):
        env.step(1)


def test_request_conservation_over_random_episodes():
    config = load_config()
    rng = np.random.default_rng(5)
    for seed in range(10):
        trace = generate_trace(seed, config)
        _, logs = rollout(config, trace, lambda env, obs: int(rng.integers(3)))
        for row in logs:
            assert row["previous_queue"] + row["arrivals"] == row["served"] + row["dropped"] + row["queue"]
        assert sum(row["arrivals"] for row in logs) == sum(row["served"] + row["dropped"] for row in logs) + logs[-1]["queue"]


def test_traces_reproducible_and_split_seeds_disjoint():
    config = load_config()
    assert generate_trace(42, config) == generate_trace(42, config)
    sets = make_traces(config)
    seed_sets = [set(trace["seed"] for trace in traces) for traces in sets.values()]
    assert all(not a.intersection(b) for i, a in enumerate(seed_sets) for b in seed_sets[i+1:])
    trace = sets["validation"][0]
    assert rollout(config, trace, lambda env, obs: 1) == rollout(config, trace, lambda env, obs: 1)


def test_baseline_threshold_boundaries_and_pending_hold():
    env = setup_env([24, 24, 24])
    assert threshold_action(env, 0.8, 0.4) == 1
    assert threshold_action(env, 0.65, 0.4) == 2
    assert fixed_action(env, 5) == 2
    env.step(2)
    assert threshold_action(env, 0.65, 0.4) == 1


def test_gymnasium_contract():
    check_env(ServiceEnvironment(), skip_render_check=True)


def test_aggregate_unserved_uses_total_arrivals():
    rows = [{"cost": 5, "arrivals": 100, "unserved": 10, "mean_queue": 1,
             "queue_above_20_pct": 0, "scaling_actions": 1, "return": -1},
            {"cost": 5, "arrivals": 900, "unserved": 0, "mean_queue": 1,
             "queue_above_20_pct": 0, "scaling_actions": 1, "return": -1}]
    assert aggregate(rows)["unserved_fraction"] == 0.01


def test_terminal_dqn_target_does_not_bootstrap():
    config = load_config()
    network, target = QNetwork(config), QNetwork(config)
    for parameter in network.parameters():
        parameter.data.zero_()
    for parameter in target.parameters():
        parameter.data.zero_()
    target.layers[-1].bias.data.fill_(100)
    optimizer = torch.optim.Adam(network.parameters(), lr=0)
    batch = (torch.zeros(2, 6), torch.zeros(2, dtype=torch.long),
             torch.tensor([-2., -2.]), torch.zeros(2, 6), torch.tensor([1., 0.]))
    loss = update_network(network, target, optimizer, batch, config)
    assert loss == pytest.approx((4 + 93**2) / 2)


def test_replay_wrap_keeps_latest_transitions():
    buffer = ReplayBuffer(2)
    for value in range(3):
        state = np.full(6, value, dtype=np.float32)
        buffer.add(state, value, -value, state + 1, False)
    states, actions, rewards, next_states, terminals = buffer.sample(np.random.default_rng(1), 2)
    assert buffer.size == 2
    assert set(actions.tolist()) == {1, 2}
    assert torch.equal(states[:, 0].long(), actions)
    assert torch.equal(rewards, -actions.float())
    assert torch.equal(next_states, states + 1)
    assert not terminals.any()

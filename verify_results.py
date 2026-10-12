import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
OUTPUT = ROOT / 'outputs'


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    episodes = read_rows(OUTPUT / 'episodes.csv')
    summary = read_rows(OUTPUT / 'summary.csv')
    assert len(episodes) == 500 and len(summary) == 10
    checks = 0
    for reported in summary:
        # Recalculate each summary from its 50 saved episode rows.
        rows = [row for row in episodes if all(row[key] == reported[key] for key in
                ['workload', 'controller', 'training_seed'])]
        assert len(rows) == 50
        assert len({row['trace_seed'] for row in rows}) == 50
        expected = {key: sum(float(row[key]) for row in rows) / len(rows) for key in
                    ['cost', 'mean_queue', 'queue_above_20_pct', 'scaling_actions', 'return']}
        expected['unserved_fraction'] = sum(float(row['unserved']) for row in rows) / sum(float(row['arrivals']) for row in rows)
        for key, value in expected.items():
            assert math.isclose(value, float(reported[key]), rel_tol=1e-12, abs_tol=1e-12)
            checks += 1
    for path in (OUTPUT / 'replay').glob('*.csv'):
        # Check request conservation, billing, terminal accounting, and reward.
        rows = read_rows(path)
        assert len(rows) == 240
        for index, row in enumerate(rows):
            values = {key: float(value) for key, value in row.items()}
            assert values['previous_queue'] + values['arrivals'] == values['served'] + values['dropped'] + values['queue']
            assert values['billed'] == values['active'] + values['pending']
            assert 1 <= values['billed'] <= 10
            assert values['terminal_queue'] == (values['queue'] if index == 239 else 0)
            reward = -(0.15 * values['billed'] + 0.04 * values['queue'] +
                       0.25 * (values['dropped'] + values['terminal_queue']) + 0.02 * values['changed'])
            assert math.isclose(reward, values['reward'], abs_tol=1e-12)
            checks += 1
    assessment = json.loads((OUTPUT / 'assessment.json').read_text())
    for workload, result in assessment.items():
        runs = [row for row in summary if row['workload'] == workload and row['controller'] == 'DQN']
        for key in result['dqn_mean']:
            values = [float(row[key]) for row in runs]
            assert math.isclose(np.mean(values), result['dqn_mean'][key], abs_tol=1e-12)
            assert math.isclose(np.std(values, ddof=1), result['dqn_sample_sd'][key], abs_tol=1e-12)
            checks += 2
    selections = json.loads((OUTPUT / 'selections.json').read_text())
    for selection in selections['dqn']:
        history = json.loads((OUTPUT / f"run_{selection['seed']}" / 'validation.json').read_text())
        expected = max(history, key=lambda row: row['validation_return'])
        assert expected['steps'] == selection['steps']
        checks += 1
    receipt = {'status': 'passed', 'numerical_checks': checks, 'episodes': len(episodes),
               'summary_rows': len(summary), 'replay_files': len(list((OUTPUT / 'replay').glob('*.csv')))}
    (OUTPUT / 'verification.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()

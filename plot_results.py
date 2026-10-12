import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from workloads import load_config

ROOT = Path(__file__).parent
OUTPUT = ROOT / 'outputs'
FIGURES = OUTPUT / 'figures'


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 12, 'text.color': 'black', 'axes.labelcolor': 'black',
                         'xtick.color': 'black', 'ytick.color': 'black'})
    config = load_config()
    styles = ['-', '--', ':']
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for seed, style in zip(config['training_seeds'], styles):
        rows = read_csv(OUTPUT / f'run_{seed}' / 'episodes.csv')
        values = np.array([float(row['return']) for row in rows])
        # This figure uses a moving mean of 20 episodes. The deck uses blocks of 40.
        smooth = np.convolve(values, np.ones(20) / 20, mode='valid')
        steps = [int(row['steps']) for row in rows][19:]
        ax.plot(steps, smooth, style, color='black', label=f'Seed {seed}')
    ax.set(xlabel='Training environment steps', ylabel='Episode return',
           title='Training return, moving mean of 20 completed episodes')
    ax.legend(frameon=False)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(FIGURES / 'training_return.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for seed, style in zip(config['training_seeds'], styles):
        rows = json.loads((OUTPUT / f'run_{seed}' / 'validation.json').read_text())
        ax.plot([row['steps'] for row in rows], [row['validation_return'] for row in rows],
                style, marker='o', color='black', label=f'Seed {seed}')
    ax.set(xlabel='Checkpoint steps', ylabel='Mean validation episode return',
           title='Validation on the same 20 traces')
    ax.legend(frameon=False)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(FIGURES / 'validation_return.png', dpi=180)
    plt.close(fig)
    summary = read_csv(OUTPUT / 'summary.csv')
    for workload in ['nominal', 'robustness']:
        rows = [row for row in summary if row['workload'] == workload]
        labels = [row['controller'] + (f" {row['training_seed']}" if row['controller'] == 'DQN' else '') for row in rows]
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        metrics = [('cost', 'Mean billed replica-intervals'),
                   ('unserved_fraction', 'Unserved requests (%)'),
                   ('queue_above_20_pct', 'Intervals with queue > 20 (%)')]
        for ax, (key, label) in zip(axes, metrics):
            # Convert only the unserved fraction to percent. Queue values already are percent.
            values = [float(row[key]) * (100 if key == 'unserved_fraction' else 1) for row in rows]
            ax.bar(range(len(rows)), values, color='white', edgecolor='black', hatch='//')
            ax.set_xticks(range(len(rows)), labels, rotation=35, ha='right')
            ax.set_ylabel(label)
            ax.grid(axis='y', alpha=0.15)
        fig.suptitle(f'{workload.capitalize()} test results, 50 shared traces')
        fig.tight_layout()
        fig.savefig(FIGURES / f'{workload}_comparison.png', dpi=180)
        plt.close(fig)
    for workload in ['nominal', 'robustness']:
        threshold = read_csv(OUTPUT / 'replay' / f'{workload}_Threshold_0.csv')
        dqn = read_csv(OUTPUT / 'replay' / f'{workload}_DQN_11.csv')
        fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
        x = np.arange(len(dqn))
        axes[0].plot(x, [float(row['arrivals']) for row in dqn], color='black', label='Arrivals')
        axes[0].set_ylabel('Requests')
        for rows, style, name in [(dqn, '-', 'DQN seed 11'), (threshold, '--', 'Threshold')]:
            axes[1].step(x, [float(row['active']) for row in rows], where='post', color='black', linestyle=style, label=name)
            axes[2].plot(x, [float(row['queue']) for row in rows], style, color='black', label=name)
            axes[3].plot(x, np.cumsum([float(row['billed']) for row in rows]), style, color='black', label=name)
        axes[1].set_ylabel('Active replicas')
        axes[2].set_ylabel('Queue')
        axes[3].set_ylabel('Cumulative cost')
        axes[3].set_xlabel('Control interval')
        for ax in axes:
            ax.legend(frameon=False, loc='upper left')
            ax.grid(alpha=0.15)
        fig.suptitle(f'First {workload} test trace, selected checkpoint for seed 11')
        fig.tight_layout()
        fig.savefig(FIGURES / f'{workload}_replay.png', dpi=160)
        plt.close(fig)
    print('Saved figures:', len(list(FIGURES.glob('*.png'))))


if __name__ == '__main__':
    main()

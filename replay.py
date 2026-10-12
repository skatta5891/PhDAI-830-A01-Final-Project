import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_results import OUTPUT, read_csv


def main(workload):
    threshold = read_csv(OUTPUT / 'replay' / f'{workload}_Threshold_0.csv')
    dqn = read_csv(OUTPUT / 'replay' / f'{workload}_DQN_11.csv')
    frames = len(dqn)
    x = np.arange(frames)
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True, dpi=120)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.87, bottom=0.08, hspace=0.22)
    fig.suptitle(f'Autoscaling replay: first {workload} test trace', fontsize=18, color='black')
    status = fig.text(0.10, 0.905, '', fontsize=11, color='black')
    arrival_values = [float(row['arrivals']) for row in dqn]
    arrival_line, = axes[0].plot([], [], color='black', marker='.', markersize=2)
    axes[0].set_ylim(0, max(arrival_values) * 1.1)
    axes[0].set_ylabel('Arrivals')
    lines = [(arrival_line, arrival_values)]
    for rows, style, name in [(dqn, '-', 'DQN seed 11'), (threshold, '--', 'Threshold')]:
        values = [[float(row['active']) for row in rows], [float(row['queue']) for row in rows],
                  np.cumsum([float(row['billed']) for row in rows])]
        for ax, y in zip(axes[1:], values):
            line, = ax.plot([], [], style, color='black', marker='.', markersize=2, label=name)
            lines.append((line, y))
    axes[1].set_ylabel('Active replicas')
    axes[1].set_ylim(0, 11)
    axes[2].set_ylabel('Queue')
    axes[2].set_ylim(0, max(25, max(float(row['queue']) for row in dqn + threshold) * 1.1))
    axes[3].set_ylabel('Replica-intervals')
    axes[3].set_ylim(0, max(sum(float(row['billed']) for row in dqn), sum(float(row['billed']) for row in threshold)) * 1.05)
    axes[3].set_xlabel('Control interval')
    cursors = [ax.axvline(0, color='gray', linestyle=':') for ax in axes]
    for ax in axes:
        ax.set_xlim(0, frames - 1)
        ax.grid(alpha=0.15)
    axes[1].legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 1.01), ncol=2)
    destination = OUTPUT / f'{workload}_demo.mp4'
    # Four intervals per second gives a 60-second replay of 240 intervals.
    with imageio.get_writer(destination, fps=4, codec='libx264', macro_block_size=16) as writer:
        for index in range(frames):
            for line, values in lines:
                line.set_data(x[:index+1], values[:index+1])
            for cursor in cursors:
                cursor.set_xdata([index, index])
            row = dqn[index]
            status.set_text(f"Interval {index}    DQN action {['remove', 'hold', 'add'][int(row['action'])]}    "
                            f"Pending replica {row['pending']}    Startup timer {row['startup_timer']}    Queue {row['queue']}")
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
    plt.close(fig)
    print(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', choices=['nominal', 'robustness'], default='nominal')
    main(parser.parse_args().workload)

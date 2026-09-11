"""Optional plots; monitoring itself does not require matplotlib."""
import argparse
import csv
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('report_dir', type=Path)
    parser.add_argument('--threshold', type=float, default=6)
    args = parser.parse_args()
    with (args.report_dir / 'scores.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    names = sorted({r['series'] for r in rows})
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 3 * len(names)), squeeze=False, sharex=True)
    for name, ax in zip(names, axes[:, 0]):
        part = [r for r in rows if r['series'] == name]
        days = [date.fromisoformat(r['date']) for r in part]
        observed = [float(r['value']) for r in part]
        expected = [float(r['expected']) for r in part]
        scale = [float(r['scale']) for r in part]
        ax.fill_between(days, [max(0, e - args.threshold * s) for e, s in zip(expected, scale)],
                        [e + args.threshold * s for e, s in zip(expected, scale)],
                        color='#dbeafe', label='Alert threshold band (not a confidence interval)')
        ax.plot(days, observed, color='#334155', linewidth=1.1, label='Observed')
        ax.plot(days, expected, color='#2563eb', linewidth=1.1, label='Expected: prior same weekdays')
        flagged = [i for i, r in enumerate(part) if r['alert'] == 'True']
        ax.scatter([days[i] for i in flagged], [observed[i] for i in flagged],
                   color='#dc2626', s=28, zorder=5, label='Unusual observation')
        ax.set_title(name.replace('_', ' '), loc='left')
        ax.set_ylabel('Daily views')
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(fontsize=8, loc='upper left')
    axes[-1, 0].set_xlabel('Date (UTC)')
    fig.suptitle('Metric Watch / Wikimedia pageviews', x=.08, ha='left', fontsize=17)
    fig.tight_layout()
    fig.savefig(args.report_dir / 'overview.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()

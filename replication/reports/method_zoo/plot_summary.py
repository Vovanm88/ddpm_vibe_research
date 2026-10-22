"""Render report-only scientific figures from frozen aggregates; no model/data calls."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
METHODS = ('cfm', 'ot_cfm', 'endpoint', 'progressive', 'reflow', 'consistency', 'shortcut', 'meanflow')
NAMES = ('CFM', 'OT-CFM', 'Endpoint', 'Progressive', 'Reflow', 'Consistency', 'Shortcut', 'MeanFlow')
COLORS = ('#64748b', '#0284c7', '#059669', '#7c3aed', '#db2777', '#d97706', '#0891b2', '#dc2626')


def main() -> None:
    ranking = json.loads((HERE / 'final/ranking.json').read_text())
    summaries = {(r['task'], r['nfe'], r['method']): r for r in ranking['summaries']}
    for task in ('rays', 'wall'):
        metrics = (('conditional_swd', 1, 'Conditional raw XY SWD ↓'),
                   ('outer_mass_gap', 100, 'Outer-coverage absolute error, pp ↓')) if task == 'rays' else (
                       ('success_fraction', 100, 'Exact all-segment success, % ↑'),
                       ('conditional_swd', 1, 'Conditional raw interior 60D SWD ↓'))
        figure, axes = plt.subplots(2, 3, figsize=(17, 10), layout='constrained')
        for column, nfe in enumerate((1, 2, 4)):
            for row, (metric, scale, title) in enumerate(metrics):
                axis = axes[row, column]
                records = [summaries[task, nfe, method] for method in METHODS]
                values = np.array([record['metrics'][metric] * scale for record in records])
                deviations = np.array([record['seed_statistics'][metric]['sample_std'] * scale for record in records])
                bars = axis.barh(np.arange(8), values, color=COLORS, alpha=.85, height=.68,
                                 xerr=deviations, error_kw={'elinewidth': 1, 'capsize': 2, 'ecolor': '#0f172a'})
                labels = [name + (' [off-target]' if method == 'endpoint' and nfe > 1 else '')
                          for name, method in zip(NAMES, METHODS, strict=True)]
                axis.set_yticks(np.arange(8), labels, fontsize=9)
                axis.invert_yaxis()
                maximum = max(float(np.max(values + deviations)), .001)
                axis.set_xlim(0, 108 if metric == 'success_fraction' else maximum * 1.22)
                axis.bar_label(bars, labels=[f'{v:.2f}' if scale == 100 else f'{v:.5f}' for v in values],
                               padding=4, fontsize=8)
                references = [reference for reference in ranking['references']
                              if reference['task'] == task and reference['nfe'] == nfe]
                for reference in references:
                    value = reference['metrics'][metric]
                    if value is not None:
                        parent = reference['league'] == 'parent_baseline'
                        axis.axvline(value * scale, color='#0f172a' if parent else '#94a3b8',
                                     linestyle='--' if parent else ':', alpha=.65, linewidth=1.1,
                                     label=('Original parent' if parent else reference['method'] + ' historical'))
                axis.grid(axis='x', alpha=.16)
                axis.set_axisbelow(True)
                axis.spines[['top', 'right']].set_visible(False)
                axis.set_title(f'Actual network NFE {nfe}\n{title}', fontsize=10)
                if row == 1:
                    axis.legend(loc='lower right', fontsize=7, framealpha=.9)
        figure.suptitle(
            ('16 conditional rotating rays' if task == 'rays' else 'Two-route conditional wall navigation')
            + ' · fixed new final bank · three training seeds\n'
            + 'Bars: mean; whiskers: sample SD across seeds. Metrics remain separate; no universal score.', fontsize=14)
        figure.supxlabel(
            'All new runs: 2000 total updates. Progressive matched exports: NFE 4 / 2 / 1 after 1200 / 1600 / 2000 cumulative updates.\n'
            'Dashed: parent at the SAME NFE. Dotted: historical M0 / E0, 1000 updates, descriptive unequal-budget controls.',
            fontsize=10)
        for extension in ('png', 'svg'):
            figure.savefig(HERE / f'{task}_summary.{extension}', dpi=150)
        plt.close(figure)


if __name__ == '__main__':
    main()

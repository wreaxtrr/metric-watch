"""Sensitivity check with frozen thresholds, separate from threshold selection."""
import json
from pathlib import Path
from monitor import classify, demo_rows, evaluate, metrics, scores


def main():
    baseline = evaluate(demo_rows())
    settings = [('different_seed', 2026, .035, 1.0),
                ('noisy_weak_incidents', 2026, .15, .3)]
    results = []
    for scenario, seed, noise, strength in settings:
        rows = demo_rows(seed, noise=noise, anomaly_strength=strength)
        for method in ('global', 'seasonal'):
            threshold = baseline[method]['threshold']
            test = [r for r in scores(rows, method) if r['split'] == 'test']
            results.append(dict(scenario=scenario, seed=seed, noise=noise,
                                anomaly_strength=strength, method=method,
                                frozen_threshold=threshold,
                                **metrics(classify(test, threshold))))
    out = Path('reports/stress')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    lines = ['# Проверка чувствительности', '',
             'Пороги зафиксированы по validation базового сценария seed=42. '
             'Здесь они не перенастраиваются. Данные полностью искусственные.', '',
             '| Сценарий | Метод | Порог | Precision | Recall | FP / 100 нормальных дней |',
             '|---|---|---:|---:|---:|---:|']
    for r in results:
        fmt = lambda x: 'n/a' if x is None else f'{x:.3f}'
        lines.append(f'| {r["scenario"]} | {r["method"]} | {r["frozen_threshold"]} | '
                     f'{fmt(r["precision"])} | {fmt(r["recall"])} | '
                     f'{fmt(r["false_alerts_per_100_normal_days"])} |')
    lines += ['', 'Чем слабее отклонение относительно шума, тем сложнее обнаружение. '
              'Нулевые ложные тревоги могут сочетаться с пропуском почти всех инцидентов. '
              'Базовый результат не является гарантией качества на другом источнике.']
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()

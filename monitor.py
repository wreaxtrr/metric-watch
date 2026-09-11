"""Daily metric monitoring. Python 3.11+, standard library only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sqlite3
import statistics as stats
from collections import defaultdict
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


def validate(rows):
    """Reject duplicates, gaps and invalid counts; missing does not mean zero."""
    grouped = defaultdict(list)
    seen = set()
    for r in rows:
        day = date.fromisoformat(r['date'])
        value = float(r['value'])
        key = (r['series'], day)
        if not r['series'] or key in seen:
            raise ValueError(f'Empty series or duplicate observation: {key}')
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid nonnegative count: {key}')
        seen.add(key)
        grouped[r['series']].append(dict(r, date=day.isoformat(), value=value))
    if not grouped:
        raise ValueError('No observations')
    for name, values in grouped.items():
        values.sort(key=lambda r: r['date'])
        for left, right in zip(values, values[1:]):
            if date.fromisoformat(right['date']) - date.fromisoformat(left['date']) != timedelta(days=1):
                raise ValueError(f'Missing daily observation in {name}: {left["date"]}')
    return [r for name in sorted(grouped) for r in grouped[name]]


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return validate(list(csv.DictReader(f)))


def scores(rows, method='seasonal', window=56):
    """One-step-ahead scoring: only observations strictly BEFORE t enter the fit."""
    if window < 14 or window % 7:
        raise ValueError('Window must contain at least two whole weeks')
    if method not in ('seasonal', 'global'):
        raise ValueError(f'Unknown method: {method}')
    rows = validate(rows)
    history = defaultdict(list)
    result = []
    for row in rows:
        past = history[row['series']]
        value = row['value']
        if len(past) >= window:
            sample = past[-window:]
            if method == 'seasonal':
                sample = sample[::7]  # window=56: same weekday as current t
                expected = stats.median(sample)
                mad = stats.median(abs(x - expected) for x in sample)
                scale = max(1.4826 * mad, 0.05 * expected, 1.0)
            elif method == 'global':
                expected = stats.mean(sample)
                scale = max(stats.pstdev(sample), 0.05 * expected, 1.0)
            else:
                raise ValueError(f'Unknown method: {method}')
            result.append(dict(row, expected=expected, scale=scale,
                               score=abs(value - expected) / scale,
                               direction='up' if value > expected else 'down'))
        past.append(value)
    return result


def classify(scored, threshold):
    return [dict(r, alert=r['score'] > threshold) for r in scored]


def metrics(rows):
    tp = sum(r['alert'] and bool(int(r['label'])) for r in rows)
    fp = sum(r['alert'] and not bool(int(r['label'])) for r in rows)
    fn = sum(not r['alert'] and bool(int(r['label'])) for r in rows)
    negatives = sum(not bool(int(r['label'])) for r in rows)
    return dict(tp=tp, fp=fp, fn=fn,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                false_alerts_per_100_normal_days=100 * fp / negatives if negatives else None)


def demo_rows(seed=42, noise=.035, anomaly_strength=1.0):
    rng = random.Random(seed)
    rows = []
    # Two validation incidents and three later held-out incidents in each series.
    incidents = [(130, 2, 2.0), (190, 4, .45), (250, 3, 1.7),
                 (290, 5, .35), (335, 2, 2.2)]
    weekday = [1.15, 1.1, 1.05, 1.0, .95, .55, .50]
    for name, level in [('docs', 5000), ('status', 800), ('help', 2000)]:
        for i in range(365):
            day = date(2025, 1, 1) + timedelta(days=i)
            value = max(0, level * weekday[day.weekday()] * (1 + .0004 * i)
                        + rng.gauss(0, noise * level))
            label, incident = 0, ''
            for start, duration, factor in incidents:
                if start <= i < start + duration:
                    value *= 1 + (factor - 1) * anomaly_strength
                    label, incident = 1, f'{name}-{start}'
            split = 'train' if i < 112 else 'validation' if i < 224 else 'test'
            rows.append(dict(series=name, date=day.isoformat(), value=round(value),
                             label=label, incident=incident, split=split))
    return rows


def evaluate(rows):
    results = {}
    for method in ['global', 'seasonal']:
        scored = scores(rows, method)
        valid = [r for r in scored if r['split'] == 'validation']
        candidates = []
        for threshold in [2, 3, 4, 5, 6, 8, 10, 15, 20]:
            m = metrics(classify(valid, threshold))
            candidates.append(dict(threshold=threshold, **m))
        feasible = [m for m in candidates if m['false_alerts_per_100_normal_days'] <= 1]
        if not feasible:
            raise ValueError('No threshold meets validation false-alarm budget')
        chosen = max(feasible, key=lambda m: (m['recall'] or 0, m['precision'] or 0, m['threshold']))
        test = classify([r for r in scored if r['split'] == 'test'], chosen['threshold'])
        incident_days = defaultdict(list)
        for r in test:
            if r['incident']:
                incident_days[r['incident']].append(r)
        detected, delays = 0, []
        for group in incident_days.values():
            found = next((i for i, r in enumerate(group) if r['alert']), None)
            if found is not None:
                detected += 1
                delays.append(found)
        results[method] = dict(threshold=chosen['threshold'], validation=chosen,
                               validation_candidates=candidates, test=metrics(test),
                               test_incidents=len(incident_days), detected_incidents=detected,
                               mean_delay_days_detected_only=stats.mean(delays) if delays else None)
    return results


def episodes(rows):
    """Consecutive same-direction alerts become one incident draft."""
    out = []
    for r in rows:
        if not r['alert']:
            continue
        previous = out[-1] if out else None
        consecutive = (previous and previous['series'] == r['series']
                       and previous['direction'] == r['direction']
                       and date.fromisoformat(r['date']) - date.fromisoformat(previous['end']) == timedelta(days=1))
        if consecutive:
            previous['end'] = r['date']
            previous['days'] += 1
            previous['max_score'] = max(previous['max_score'], r['score'])
        else:
            out.append(dict(series=r['series'], start=r['date'], end=r['date'],
                            direction=r['direction'], days=1, max_score=r['score']))
    return out


def save_database(path, rows, scored, model_key):
    """Input upsert; replace the current run's scored snapshot and draft outbox."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS observations (
          series TEXT, day TEXT, value REAL NOT NULL CHECK(value >= 0),
          PRIMARY KEY(series, day));
        CREATE TABLE IF NOT EXISTS scores (
          model TEXT, series TEXT, day TEXT, expected REAL, value REAL,
          score REAL, alert INTEGER, PRIMARY KEY(model, series, day));
        CREATE TABLE IF NOT EXISTS alert_drafts (
          id TEXT PRIMARY KEY, model TEXT, series TEXT, start TEXT, end TEXT,
          direction TEXT, days INTEGER, max_score REAL);
        ''')
        db.executemany('INSERT INTO observations VALUES (?, ?, ?) ON CONFLICT(series,day) DO UPDATE SET value=excluded.value',
                       [(r['series'], r['date'], r['value']) for r in rows])
        db.execute('DELETE FROM scores WHERE model=?', (model_key,))
        db.execute('DELETE FROM alert_drafts WHERE model=?', (model_key,))
        db.executemany('INSERT INTO scores VALUES (?, ?, ?, ?, ?, ?, ?)',
                       [(model_key, r['series'], r['date'], r['expected'], r['value'], r['score'], int(r['alert'])) for r in scored])
        for e in episodes(scored):
            key = hashlib.sha256(f'{model_key}|{e["series"]}|{e["start"]}|{e["direction"]}'.encode()).hexdigest()[:20]
            db.execute('INSERT INTO alert_drafts VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                       (key, model_key, e['series'], e['start'], e['end'], e['direction'], e['days'], e['max_score']))


def report(out, scored, threshold, source, evaluation=None):
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'scores.csv', scored)
    draft = episodes(scored)
    (out / 'alert_drafts.json').write_text(json.dumps(draft, indent=2, ensure_ascii=False), encoding='utf-8')
    text = ['# Metric Watch — результат запуска', '', f'Источник: {source}', '',
            f'Порог: {threshold}. Проверено наблюдений после прогрева: {len(scored)}. '
            f'Дней с отклонениями: {sum(r["alert"] for r in scored)}. Эпизодов: {len(draft)}.', '',
            'Алерт означает необычное значение, а не доказанный сбой или причинный эффект. '
            'Сообщения никуда не отправлены; alert_drafts.json — локальные черновики.', '',
            '| Ряд | Начало | Конец | Направление | Дней | Максимальный score |',
            '|---|---|---|---|---:|---:|']
    for e in draft:
        name = e['series'].replace('|', '\\|').replace('\n', ' ')
        text.append(f'| {name} | {e["start"]} | {e["end"]} | {e["direction"]} | {e["days"]} | {e["max_score"]:.2f} |')
    if evaluation:
        text += ['', '## Отложенная проверка на синтетических инцидентах', '',
                 'Порог выбирается на validation при бюджете ≤1 ложной тревоги на 100 нормальных дней. '
                 'Test не участвует в выборе. Метрики ниже относятся только к синтетике.', '',
                 '| Метод | Порог | Precision | Recall | FP / 100 нормальных дней | Найдено инцидентов |',
                 '|---|---:|---:|---:|---:|---:|']
        for method, item in evaluation.items():
            m = item['test']
            fmt = lambda x: 'n/a' if x is None else f'{x:.3f}'
            text.append(f'| {method} | {item["threshold"]} | {fmt(m["precision"])} | {fmt(m["recall"])} | '
                        f'{fmt(m["false_alerts_per_100_normal_days"])} | {item["detected_incidents"]}/{item["test_incidents"]} |')
        (out / 'evaluation.json').write_text(json.dumps(evaluation, indent=2), encoding='utf-8')
    else:
        text += ['', '## Ограничения', '',
                 'На реальных рядах нет разметки инцидентов: precision, recall и ложные тревоги неизвестны. '
                 'Порог из синтетического эксперимента — стартовая настройка, его переносимость не доказана. '
                 'Проверьте новости, праздники, изменение состава трафика и качество загрузки перед выводами.']
    (out / 'report.md').write_text('\n'.join(text) + '\n', encoding='utf-8')


def run(rows, out, threshold, source, evaluation=None):
    rows = validate(rows)
    scored = classify(scores(rows), threshold)
    if not scored:
        raise ValueError('Need at least 57 consecutive daily observations per series to score')
    model = f'seasonal-mad-v1-w56-k{threshold:g}'
    save_database(out / 'monitor.sqlite', rows, scored, model)
    report(out, scored, threshold, source, evaluation)
    print(json.dumps(dict(report=str(out / 'report.md'), observations=len(rows),
                          scored=len(scored), alert_days=sum(r['alert'] for r in scored),
                          episodes=len(episodes(scored))), ensure_ascii=False))


def fetch(args):
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    if start > end or end >= datetime.now(timezone.utc).date():
        raise ValueError('Use a valid range ending before the current UTC day')
    rows, sources = [], []
    for article in dict.fromkeys(args.articles):
        article = article.replace(' ', '_')
        url = ('https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/'
               f'en.wikipedia.org/all-access/user/{quote(article, safe="")}/daily/'
               f'{start:%Y%m%d}00/{end:%Y%m%d}00')
        with urlopen(Request(url, headers={'User-Agent': args.user_agent}), timeout=30) as response:
            payload = json.load(response)
        part = [dict(series=article, date=datetime.strptime(r['timestamp'], '%Y%m%d%H').date().isoformat(),
                     value=r['views']) for r in payload['items']]
        part = validate(part)
        if part[0]['date'] != start.isoformat() or part[-1]['date'] != end.isoformat():
            raise ValueError(f'Incomplete requested date range for {article}')
        rows.extend(part)
        sources.append(dict(url=url, article=article, rows=len(part)))
    write_csv(args.output, validate(rows))
    metadata = dict(retrieved_at=datetime.now(timezone.utc).isoformat(), timezone='UTC',
                    project='en.wikipedia.org', agent='user', access='all-access', sources=sources)
    Path(args.output).with_suffix('.source.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Wrote {len(rows)} observations to {args.output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    demo = subs.add_parser('demo', help='Seeded synthetic benchmark; works offline')
    demo.add_argument('--output', type=Path, default=Path('reports/demo'))
    demo.add_argument('--seed', type=int, default=42)
    load = subs.add_parser('fetch', help='Download historical Wikimedia pageviews')
    load.add_argument('--articles', nargs='+', default=['Python_(programming_language)', 'SQL', 'Statistics'])
    load.add_argument('--start', default='2025-01-01')
    load.add_argument('--end', default='2025-06-30')
    load.add_argument('--user-agent', required=True, help='Descriptive client name and your contact URL')
    load.add_argument('--output', type=Path, default=Path('data/wikimedia.csv'))
    scan = subs.add_parser('run', help='Replay a daily CSV in chronological order')
    scan.add_argument('input', type=Path)
    scan.add_argument('--output', type=Path, default=Path('reports/wikimedia'))
    scan.add_argument('--threshold', type=float, default=6)
    args = parser.parse_args()
    if args.command == 'demo':
        rows = demo_rows(args.seed)
        evaluation = evaluate(rows)
        write_csv(args.output / 'synthetic.csv', rows)
        # Full history is exported; held-out quality is computed separately above.
        run(rows, args.output, evaluation['seasonal']['threshold'],
            f'Synthetic benchmark, seed={args.seed}; not production data', evaluation)
    elif args.command == 'fetch':
        fetch(args)
    else:
        if not math.isfinite(args.threshold) or args.threshold <= 0:
            parser.error('threshold must be finite and positive')
        run(read_csv(args.input), args.output, args.threshold, str(args.input))


if __name__ == '__main__':
    main()

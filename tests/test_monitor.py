import copy
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from monitor import classify, demo_rows, episodes, evaluate, metrics, save_database, scores, validate


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.rows = demo_rows()

    def test_future_cannot_change_past_scores(self):
        original = scores(self.rows)
        changed = copy.deepcopy(self.rows)
        for row in changed:
            if row['date'] >= '2025-10-01':
                row['value'] *= 100
        before = [r for r in original if r['date'] < '2025-10-01']
        after = [r for r in scores(changed) if r['date'] < '2025-10-01']
        self.assertEqual(before, after)

    def test_current_observation_not_in_its_own_forecast(self):
        changed = copy.deepcopy(self.rows)
        changed[100]['value'] *= 100
        a = {(r['series'], r['date']): r for r in scores(self.rows)}
        b = {(r['series'], r['date']): r for r in scores(changed)}
        key = changed[100]['series'], changed[100]['date']
        self.assertEqual(a[key]['expected'], b[key]['expected'])
        self.assertEqual(a[key]['scale'], b[key]['scale'])

    def test_test_values_do_not_change_selected_threshold(self):
        changed = copy.deepcopy(self.rows)
        for r in changed:
            if r['split'] == 'test':
                r['value'] *= 10
        a, b = evaluate(self.rows), evaluate(changed)
        self.assertEqual([x['threshold'] for x in a.values()], [x['threshold'] for x in b.values()])

    def test_rejects_duplicates_gaps_negative_nan(self):
        for bad in [self.rows + [self.rows[0]], self.rows[:20] + self.rows[21:]]:
            with self.assertRaises(ValueError):
                validate(bad)
        for value in [-1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                validate([dict(self.rows[0], value=value)])

    def test_seasonality_without_anomaly_does_not_alert(self):
        rows = copy.deepcopy(self.rows[:365])
        # Fixed weekly pattern with large weekend drop; not anomalous.
        from datetime import date
        for r in rows:
            r['value'] = 100 if date.fromisoformat(r['date']).weekday() < 5 else 20
        self.assertFalse(any(r['alert'] for r in classify(scores(rows), 3)))

    def test_confusion_matrix_denominators(self):
        result = metrics([dict(alert=True, label=1), dict(alert=True, label=0),
                          dict(alert=False, label=1), dict(alert=False, label=0)])
        self.assertEqual(result['precision'], .5)
        self.assertEqual(result['recall'], .5)
        self.assertEqual(result['false_alerts_per_100_normal_days'], 50)
        self.assertIsNone(metrics([dict(alert=False, label=0)])['precision'])

    def test_repeated_run_does_not_duplicate_records(self):
        rows = validate(self.rows)
        scored = classify(scores(rows), 6)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'test.sqlite'
            save_database(path, rows, scored, 'test')
            save_database(path, rows, scored, 'test')
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM observations').fetchone()[0], len(rows))
                self.assertEqual(db.execute('SELECT COUNT(*) FROM alert_drafts').fetchone()[0], len(episodes(scored)))

    def test_consecutive_alerts_group_but_opposite_direction_does_not(self):
        base = dict(series='a', alert=True, score=8, direction='up')
        rows = [dict(base, date='2025-01-01'), dict(base, date='2025-01-02'),
                dict(base, date='2025-01-03', direction='down')]
        self.assertEqual([e['days'] for e in episodes(rows)], [2, 1])


if __name__ == '__main__':
    unittest.main()

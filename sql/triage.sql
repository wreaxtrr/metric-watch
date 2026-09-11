-- Read the same database written by monitor.py run/demo.
-- This ranks unusual episodes for manual review, not business losses.
SELECT series, start, end, direction, days, ROUND(max_score, 2) AS severity
FROM alert_drafts
ORDER BY max_score DESC, start DESC;

-- Daily alert share is a workload indicator, NOT a false-positive rate.
SELECT model, series, COUNT(*) AS scored_days, SUM(alert) AS alerted_days,
       ROUND(100.0 * SUM(alert) / COUNT(*), 2) AS alert_day_percent
FROM scores
GROUP BY model, series;

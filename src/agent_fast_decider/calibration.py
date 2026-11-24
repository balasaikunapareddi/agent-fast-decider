"""Calibration constants for :class:`FastLocalDecider`.

Fit on the dev half of the benchmark set (tickets T01-T20) by
grid search maximising log-likelihood of the gold labels
(see ``scripts/fit_calibration.py``). Dev accuracy at fit:
route 20/20, urgent 20/20, frustration 14/20. The benchmark reports held-out
accuracy on T21-T40 separately, so the fit cannot flatter the
headline numbers. Re-run the fit script if the lexicons or the
ticket set change.
"""

ROUTE_TEMPERATURE = 0.5

URGENCY_SLOPE = 4.0
URGENCY_INTERCEPT = -3.9

FRUSTRATION_TEMPERATURE = 0.5

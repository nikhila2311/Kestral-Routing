# Test predictions: summary

Model `logreg_c1_none-918025186907`, fitted on all 10,822 labelled requests (2025-04-01 to 2026-06-30). 2,178 test requests (2026-07-01 to 2026-09-30), predicted through the saved router, as the API does.

This model has not been scored: no labels after its training window are available. The validation accuracy of 83.89% belongs to the same configuration fitted on the train window only.

## Predicted volume per team

Auto-routed requests go straight to the team. For triaged requests the team is only the model's first guess: on validation it was right for 28.93% of them (auto-routed: 96.32%). The auto-routed column is the reliable part; the triaged requests will be spread over the teams in a way that is not yet known.

| Team | Total | Share | Per month | Auto-routed | Triaged (first guess) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Repairs | 604 | 27.73% | 201.3 | 447 | 157 |
| Installs & Demo | 316 | 14.51% | 105.3 | 264 | 52 |
| Returns & Replacement | 290 | 13.31% | 96.7 | 253 | 37 |
| Warranty Claims | 265 | 12.17% | 88.3 | 216 | 49 |
| Product Advice | 259 | 11.89% | 86.3 | 219 | 40 |
| Billing | 251 | 11.52% | 83.7 | 213 | 38 |
| Filters & Consumables | 193 | 8.86% | 64.3 | 171 | 22 |

## Auto-routed requests only

Each team counts only the requests sent straight to it; shares are of all auto-routed requests. Triaged requests are one separate count because their team is not yet known. They will still reach some team, so each team's count is not its full workload.

| Team | Auto-routed | Share | Per month | 2026-07 | 2026-08 | 2026-09 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Repairs | 447 | 25.07% | 149.0 | 158 | 157 | 132 |
| Installs & Demo | 264 | 14.81% | 88.0 | 100 | 83 | 81 |
| Returns & Replacement | 253 | 14.19% | 84.3 | 88 | 80 | 85 |
| Product Advice | 219 | 12.28% | 73.0 | 82 | 62 | 75 |
| Warranty Claims | 216 | 12.11% | 72.0 | 61 | 80 | 75 |
| Billing | 213 | 11.95% | 71.0 | 74 | 81 | 58 |
| Filters & Consumables | 171 | 9.59% | 57.0 | 48 | 60 | 63 |
| All auto-routed | 1,783 | | 594.3 | 611 | 603 | 569 |
| Triaged, team not yet known | 395 | | 131.7 | 125 | 138 | 132 |

Busiest by total predicted volume: Repairs, then Installs & Demo and Returns & Replacement. Read with the bias table below.

## By month (total, of which triaged)

| Team | 2026-07 | 2026-08 | 2026-09 |
| --- | ---: | ---: | ---: |
| Repairs | 193 (35) | 222 (65) | 189 (57) |
| Installs & Demo | 126 (26) | 95 (12) | 95 (14) |
| Returns & Replacement | 100 (12) | 90 (10) | 100 (15) |
| Warranty Claims | 76 (15) | 95 (15) | 94 (19) |
| Product Advice | 96 (14) | 78 (16) | 85 (10) |
| Billing | 90 (16) | 95 (14) | 66 (8) |
| Filters & Consumables | 55 (7) | 66 (6) | 72 (9) |
| All requests | 736 | 741 | 701 |
| Triage share | 16.98% | 18.62% | 18.83% |

## Triage and vague requests

Triaged: 395 requests, 18.14% (sanity band 10.00% to 30.00%). Confidence threshold 0.6283, frozen before this model was fitted.

Vague (rule-based estimate): 177 requests, 8.13%; 177 of them are triaged.

## Compared with the last labelled months

Closing team of 2,135 requests, 2026-04-01 to 2026-06-30 (3 months). Largest gap: Repairs, 4.41 points (limit 10).

| Team | Predicted share | Recent share | Gap, points | Recent per month |
| --- | ---: | ---: | ---: | ---: |
| Repairs | 27.73% | 23.33% | +4.41 | 166.0 |
| Installs & Demo | 14.51% | 15.04% | -0.53 | 107.0 |
| Returns & Replacement | 13.31% | 15.27% | -1.95 | 108.7 |
| Warranty Claims | 12.17% | 12.13% | +0.04 | 86.3 |
| Product Advice | 11.89% | 12.60% | -0.71 | 89.7 |
| Billing | 11.52% | 11.94% | -0.42 | 85.0 |
| Filters & Consumables | 8.86% | 9.70% | -0.83 | 69.0 |

## Known bias: do not read the volumes as headcount

On the 2,135 validation requests the same configuration predicted some teams too often, so predicted volumes are not unbiased. No correction is applied: the bias on the test months is not known.

| Team | Predicted on validation | Closed by the team | Difference, points |
| --- | ---: | ---: | ---: |
| Repairs | 582 | 498 | +3.93 |
| Installs & Demo | 308 | 321 | -0.61 |
| Returns & Replacement | 307 | 326 | -0.89 |
| Warranty Claims | 253 | 259 | -0.28 |
| Product Advice | 239 | 269 | -1.41 |
| Billing | 254 | 255 | -0.05 |
| Filters & Consumables | 192 | 207 | -0.70 |

Notes: the predicted team is always the model's first choice, with no rule or lookup on top. Some test messages also appear, word for word, in the training data. Predictions are reproducible byte for byte on one machine with the pinned library versions; another processor or maths library could flip a near-tie.

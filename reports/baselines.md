# Baselines

Trained on 8,687 requests and scored against the final team on 2,135 validation requests (2026-04-01 to 2026-06-30, 3 months).
Costs are for misroutes only and leave out the bot licence.

| Baseline | Accuracy | 95% interval | Macro F-score | Agreement with bot label | Misroutes | Cost (Rs) | Cost per month (Rs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| (a) Majority team | 23.33% | 21.55% to 25.15% | 5.40% | 28.90% | 1,637 | 1,142,723 | 380,908 |
| (b) Bot label | 76.67% | 74.89% to 78.36% | 77.95% | 100.00% | 498 | 347,634 | 115,878 |
| (c) Keyword rules | 65.95% | 64.07% to 67.96% | 65.30% | 68.62% | 727 | 507,489 | 169,163 |
| (d) TF-IDF, trained on the final team | 83.89% | 82.30% to 85.39% | 83.98% | 80.52% | 344 | 240,132 | 80,044 |
| (e) TF-IDF, trained on the bot label | 76.21% | 74.47% to 77.99% | 77.35% | 95.64% | 508 | 354,614 | 118,205 |

## Cost sensitivity (Rs per month)

| Cost per misroute (Rs) | (a) Majority team | (b) Bot label | (c) Keyword rules | (d) TF-IDF, trained on the final team | (e) TF-IDF, trained on the bot label | Saving, (d) against (b) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 698.06 | 380,908 | 115,878 | 169,163 | 80,044 | 118,205 | 35,834 |
| 565.00 | 308,302 | 93,790 | 136,918 | 64,787 | 95,673 | 29,003 |

- Our misroutes are assumed to bounce between teams as often as the bot's did.
- The Rs 698.06 average covers all labelled months, including the validation months, so it is a reporting constant and must not be used to choose thresholds.
- Rs 565.00 assumes a single transfer per misroute.

## Repeated text

- Validation texts also in the train window: 9.98% (213 seen, 1,922 unseen).
- Test texts also in the train window: 10.15%.

| Baseline | Accuracy, seen text | Accuracy, unseen text |
| --- | ---: | ---: |
| (b) Bot label | 80.28% | 76.27% |
| (d) TF-IDF, trained on the final team | 84.98% | 83.77% |

## Findings

- TF-IDF trained on the final team reaches 83.89%, 7.21 points above the bot's 76.67%; their 95% intervals do not overlap. On the same rows, the paired 95% interval for the difference is 5.76 to 8.81 points.
- The same pipeline trained on the bot's labels agrees with the bot on 95.64% of requests (against 80.52%) but matches the final team on 76.21%; the model trained on the final team is 7.68 points above that.
- The keyword rules reach 65.95%, 10.73 points below the bot; a rule fires on 61.69% of requests and the rest go to Repairs.
- At Rs 698.06 per misroute, the bot's errors cost Rs 115,878 per month on validation and the TF-IDF model's Rs 80,044, a monthly saving of Rs 35,834, before the bot's licence of Rs 26,667 per month.
- Repeated messages are resampled as if independent, so the intervals are slightly too narrow.

# Router

Model `logreg_c1_none-37d79393afea` (scikit-learn 1.9.1, joblib 1.6.0, numpy 2.5.5), fitted on 8,687 requests (2025-04-01 to 2026-03-31). Scored on 2,135 validation requests (2026-04-01 to 2026-06-30). Reporting only: nothing was tuned on these numbers.

## Triage

A request goes to triage when the expected misroute cost exceeds one clarifying contact: (1 - confidence) x Rs 699.49 > Rs 260, that is, confidence below 0.6283. Messages without text always go to triage. The misroute cost is a train-window average. Triage never changes the team.

| | Requests | Accuracy | Misroutes |
| --- | ---: | ---: | ---: |
| Sent straight on | 1,741 | 96.32% | 64 |
| Triaged | 394 | 28.93% | 280 |
| All | 2,135 | 83.89% | 344 |
| Bot, all | 2,135 | 76.67% | 498 |

- Triage rate: 18.45%. Mean confidence of triaged requests: 32.96%, against 28.93% accuracy.
- Vague requests (M2 rule): 173 in validation, 173 of them triaged.
- Repairs precision: 97.35% on the 415 requests sent straight to Repairs, 77.49% on all.

## Cost per month, Rs

Misroutes sent straight on x Rs 698.06 + triaged requests x Rs 260, over 3 months. **Assumption: a clarified request is never misrouted.** That is the best case; the other rows show what it is worth.

| | Per month |
| --- | ---: |
| Router with triage (headline, best case) | 49,039 |
| Router with triage, clarification fixes nothing | 114,191 |
| Router without triage | 80,044 |
| Bot, same requests | 115,878 |
| Bot licence, not included above | 26,667 |

- Headline saving against the bot: Rs 66,839 per month.
- Break-even clarification rate: 52.41%. Triage beats no triage only if clarification fixes more than this share of the triaged misroutes. Formula: triaged requests x cost_clarify / (triaged misroutes x cost_per_misroute_reporting) = 394 x 260 / (280 x 698.06).

## Policy §3 check (invented messages)

| Message | Team | Confidence | Triage | §3 warning |
| --- | --- | ---: | --- | --- |
| I paid for installation but technician never came | Installs & Demo | 99.10% | no | no |
| I have paid the full amount and the filter has not been delivered | Installs & Demo | 36.37% | yes | no |
| Paid by card last week, now the heater is not working | Repairs | 90.18% | no | no |
| I was charged twice for the same order | Billing | 97.97% | no | no |
| Please convert my payment to EMI and send the GST invoice | Billing | 85.33% | no | no |

The team is whatever the model says; no rule overrides it. Channel, product and warranty were left out.

## Caveats

- The model is the M4 baseline (d), whose validation score had already been seen, so this is not an untouched estimate.
- The cost per misroute used for reporting covers all labelled months, validation included, and assumes our misroutes bounce as often as the bot's did.
- Triage depends on the probabilities being honest; they are not recalibrated here.
- Reasons quote words with positive weight for the predicted team. The model also uses character patterns and the request fields, which are not shown, so reasons are a partial explanation.
- Clarifying questions and reasons are fixed wording, untested with customers and agents.

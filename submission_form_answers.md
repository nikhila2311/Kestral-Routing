# Submission form answers

## Expected score

**83.9% accuracy, likely range 80.7% to 86.0%** (macro-F1 about 0.840), if the hidden labels are the team that closed each
request.

- The figure comes from a time-based check: trained on April 2025 to March 2026, scored on 2,135 requests from April to
  June 2026. The bot gets 76.7% on the same requests, so the lead is 7.2 points (paired 95% interval 5.8 to 8.8).
- The range uses the bootstrap interval (82.3% to 85.4%), widened because the test is another sample, with the low end
  moved down 0.9 points for drift between windows.
- I lean to the lower half. Accuracy fell each month (85.1%, 84.3%, 82.2%); the bot's fell too (78.9%, 77.0%, 74.1%).
- **If the hidden labels are the bot's:** about 80.5% (range 78.2% to 83.0%, macro-F1 about 0.810), because the router
  often disagrees with the bot where the bot was wrong. I did not optimise for this case: a model trained on bot labels
  agrees with them 95.6% of the time but is right on only 76.2%.
- A rough ceiling is 93.6%: about 8.1% of requests are too vague to route from the text.

## Approach summary

- Clean the text (repair damaged characters, map old team names), then train on the team that closed the request.
- Model: word and character TF-IDF plus channel, product and warranty status, into logistic regression. It runs on CPU,
  with a fixed seed and no paid service.
- Triage: below a confidence threshold of 0.628, derived from the misroute cost in the training window, the request gets
  one clarifying question. On validation 81.5% are sent straight on at 96.3%; the first guess on the 18.5% flagged would
  be right only 28.9% of the time.
- The final model is refitted on all 10,822 labelled requests. `predictions.csv` has 2,178 rows; flagged requests carry
  the model's first choice.
- A FastAPI service and one page show the team, reasons and question.

## Decisions on unclear points

- **Target:** the team that closed the request, not the bot's label, which agrees with it on 77.2% of requests.
- **The 90% bar:** it is stated against the bot's labels, so I restated it against the closing team.
- **Renames:** old names are mapped to the current seven when the data is loaded.
- **Zoho timestamps:** legacy close times are in UTC, so they are moved to IST before any duration is computed.
- **Vague requests:** asked, not guessed. All 173 on validation and all 177 in the test set, by my word-list rule, are
  flagged.
- **"I paid":** Billing only when the payment itself is the problem.
- **No LLM in the product:** no per-request cost, no evidence of better accuracy, and customer data must not be shared.
- **Costs:** the headline Rs 35,834 a month saved assumes no clarifying question is asked. With the question it ranges
  from Rs 1,687 (never helps) to Rs 66,839 (always works); break-even is 52.4% of flagged requests whose first guess
  was wrong.
- **Labelled months:** the email says eighteen; the labelled data covers 15.
- **Validation already seen:** my first selection rule picked a model that scored 82.6% on validation, 1.3 points below
  the baseline and outside the allowed 1.0. I amended the rule after seeing that. The model it then chose had already been
  scored, so 83.9% is not an untouched estimate.

## What I tried, changed and threw away

- **Tried:** the bot, keyword rules (65.9%), models trained on the bot's labels and on the closing team, and 22 candidate
  configurations compared on January to March 2026.
- **Changed:** the target, the selection rule (to favour trustworthy confidence) and the triage threshold (training window
  only). One keyword rule was corrected after validation had been seen. Question and reason wording was revised.
- **Threw away:** copying the bot, added keyword features (none met the log loss limit), class balancing, calibrated
  linear SVMs (up to 0.7 points higher on the inner check, but more complex), correcting predicted volumes, and an LLM.

## AI tools used

- **Claude (chat on claude.ai):** planning, module design, writing prompts and reviewing outputs.
- **Claude Code:** implementation, tests and code reviews.
- No LLM runs inside the product.
- Approximate cost: [YOUR COST]

## Known limitations

- The final model has no held-out score; the 83.9% belongs to the version trained to March 2026.
- Nobody knows how often a clarifying question works. If never, the router saves almost nothing over the bot.
- Repairs is over-predicted by 3.9 points on validation, so predicted volumes are not headcount.
- The bot is more precise on four teams: Installs & Demo, Returns & Replacement, Warranty Claims and Product Advice.
- Of the router's 344 validation errors, 230 are vague requests or close calls.
- Confidence is too high in the 0.2 to 0.5 range.
- Savings depend on the cost per misroute: at Rs 565 instead of Rs 698, the saving is Rs 29,003 a month.
- About 10.1% of test texts also appear in the training window; validation is similar (10.0%).
- The service has no login and is meant for local use.

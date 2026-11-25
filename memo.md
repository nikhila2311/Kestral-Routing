# Memo: replacing the routing bot

**To:** Ritu Deshpande, Head of D2C Operations

## The decision

Switch off the bot, after a two-week shadow run with the new router beside it. Proposed conditions, on requests that have
closed: the router is right at least 3 points more often than the bot; at least 93% of requests it sends straight on are
right; and Meenal agrees.

## The number

"90% match to the old labels" is the wrong bar: those labels are the bot's own guesses. Against the team that actually
closed each request, the bot was right 77.2% of the time over 15 months. We built a copy of the bot: it matches the labels
95.6% of the time, so it passes your bar, yet is right only 76.2% of the time.

On April–June 2026 the router was right on 83.9% of requests, the bot on 76.7%. Expect 80.7% to 86.0% on new requests.
The router sends 81.5% straight on, 96.3% of them correctly. The other 18.5% get one clarifying question instead of a
guess, including all 173 vague "please call me" requests (by our word-list rule).

## The rupees

| Cost of wrong routing, April–June 2026 | Rs per month |
|---|---:|
| Bot | 115,878 |
| Router, no question asked | 80,044 |
| Router, question always works | 49,039 |
| Router, question never helps | 114,191 |

**Plan on saving Rs 35,834 a month, which assumes no question is asked.** The question could lift that to Rs 66,839, but
each costs Rs 260: if it never helped, the saving would be Rs 1,687. It pays only if it fixes over 52.4% of flagged
requests whose first guess was wrong.

Separately, dropping the bot licence saves Rs 3.2 lakh a year.

For Farhan: no per-request AI charges; one small ordinary server. Estimate, to confirm with Tanmay: Rs 2,000–4,000 a month,
plus one day's retraining each quarter.

## Next week

- **Shadow run:** start it beside the bot; decide after two weeks.
- **Question trial:** Meenal's team asks the question on flagged requests (about 132 a month) and records the first guess
  and the closing team. Two weeks is too few; judge after two months and stop if it misses 52.4%.
- **Headcount:** plan on where requests actually closed: Repairs 166 a month, Returns 109, Installs 107 are busiest. Bot
  labels overstate Repairs by 23%, Billing by 26% and Filters by 37%; they understate Installs, Returns and Warranty by
  24–26% and Product Advice by 15%.

**Data:** the router keeps customer data inside Kestrel's systems (policy §10) and uses no outside AI service.

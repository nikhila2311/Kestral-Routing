# Recording script (under 3 minutes)

## What I tried

**[SHOW: reports/figures/confusion_matrices.png]**

Kestrel wants to replace its routing bot. The obvious plan was to learn from the bot's labels. So first I checked them
against the team that actually closed each request. The bot was right about 77% of the time.

I tried four things: the bot, keyword rules, a model trained on the bot's labels, and the same model trained on the
closing team. On April to June 2026 the bot scored 76.7% and its copy 76.2%. The model trained on the closing team scored
83.9%. Same model, different target. These grids show where the bot and my router go wrong.

## What I changed

**[SHOW: reports/figures/coverage_accuracy.png]**

The biggest change was the target. The second was admitting that some requests cannot be routed from the text. When the
router is not confident, it asks one question instead of guessing. It sends 81.5% straight on, and 96.3% of those are
right.

**[SHOW: the service, click "Paid, but not a payment problem"]**

This customer mentions paying, but the problem is a missing technician. It goes to Installs and Demo, not Billing, and
the screen says why.

**[SHOW: type "Is it normal that my cooktop makes a humming sound"]**

A question or a fault? The first guess is Product Advice, with Repairs close behind. The router is not sure, so it asks.

**[SHOW: click "Vague request"]**

Here nothing says which team. Every vague request in April to June was caught.

I also changed how I picked the model. My first rule chose one that did worse on later months, so I amended it. I say
so openly: the final score is not an untouched estimate.

## What I threw away

**[SHOW: memo.md, "The rupees"]**

I threw away copying the bot. It matches the old labels 95.6% of the time, which passes the 90% bar, and it is no better
than the bot. I threw away hand-made keyword features, because none met my limits. And I left out an LLM: no per-request
charges, and customer data stays inside Kestrel's systems.

I also threw away the best-case saving as the headline. The safe figure is Rs 35,834 a month, with no question asked.
The question adds more only if it works more than half the time. So: a shadow run first, then switch off the bot.

# address-linkage

Link people by where they live: the same address, the same house, the same building or complex, and
neighbours. It also flags one person stored as two records (for example two admissions).

It reuses the Amazon ML Challenge entity-resolution pipeline (`../ml-challenge`: blocking, dense
retrieval, staged LightGBM) and feeds it home addresses only.

## Privacy

- Everything the tool writes goes to `work/` (git-ignored). Those files hold names and home addresses:
  keep them on this machine, do not publish or upload them.
- The model reads addresses only. Relation phrases inside addresses ("S/O <name>", "C/O <name>") are
  removed before modelling, so no family or guardian name reaches it.
- Names, dates of birth and parents' names are used only after scoring: to label same-person pairs in
  the report and to evaluate the output.
- Logs carry counts and ids, never names or addresses.

## Setup

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

- The ml-challenge repo must sit next to this one, or set `ML_CHALLENGE_SRC=/path/to/ml-challenge/src`.
  Its `run.py` needs the `exclude_same_id` option (matching a list against itself).
- Database access comes from `DB_HOST`, `DB_USER`, `DB_NAME` and `MYSQL_PWD` (defaults: local root, `mis_40_50`).
- Outputs go to `work/`, or set `LINKAGE_WORK`.

## Usage

```sh
python -m address_linkage all          # export, train, score, evaluate, report (about 15 min on a laptop CPU)
python -m address_linkage <step>       # export | train | score | evaluate | report
python -m address_linkage gold-sample  # draw the gold set
python -m address_linkage gold-page    # blind labelling page: work/gold/gold_labelling.html
python -m address_linkage gold-score gold_labels.csv   # leaderboard score from your labels
```

Run on mains power with Low Power Mode off, and on macOS under `caffeinate -i` (for example
`caffeinate -i python -m address_linkage all`): a throttled or sleeping machine turns a 15-minute run into hours.

Open `work/report.html` for the results: overview, people with their ranked neighbours and the evidence
for each, pairs to review, households and duplicate records.

## How it works

1. **export** reads permanent addresses (present addresses are mostly hostel or campus, so they are
   skipped), cleans them (lower case, relation phrases and pincode removed, renamed cities and Hindi
   synonyms unified, repeated parts removed, spelling variants merged within a pincode), and writes the
   challenge input format: every person is both an S1 record and a query, scoped by pincode.
2. **Labels** come from address-only rules (`rules.py`): exact specific address, same house number on the
   same street, one address containing the other, a rare place name shared in one pincode, a shared
   landmark. 20% of pincodes are held out of training.
3. **train** runs the full challenge pipeline: fine-tuned E5-small encoder, lexical + dense blocking,
   Stage A pruning, Stage B/C LightGBM.
4. **score** takes Stage C's pair probability and raises it where a rule is certain (identical specific
   address 1.0, same house 0.95, more detail 0.9). Identical text shared by more than six people (campus,
   hostel) is marked as a generic address, not a household. A matching name plus a matching date of birth
   or father's name marks one person with two records.

## Evaluation

There is no ground truth for "neighbour", so two independent checks are used:

- **Family answer key** (`evaluate`): siblings (both parents' names match, different date of birth, same
  pincode) should share an address. It measures recall and whether a sibling is ranked as a person's #1
  neighbour, with 95% intervals from a bootstrap over pincodes. The family share of output pairs is a
  lower bound on precision.
- **Gold set** (`gold-*`): a stratified sample of 150 people whose candidates a person labels blind
  (same house / neighbour / not near). It is scored like the challenge leaderboard: F0.5 per person,
  where a person with nobody nearby scores 1 only if nothing is predicted, averaged with stratum weights.
  The decoding policy (threshold, or the challenge's expected-F0.5 decoder on calibrated scores) is
  chosen on the dev half and reported on the test half.

## Results

From `python -m address_linkage all` on 34,874 permanent addresses (8,184 pincodes). The answer key has 221
sibling pairs in one pincode, 54 of them in held-out pincodes; 95% intervals in brackets.

| Measure | All pincodes | Held-out pincodes |
|---|---|---|
| Sibling ranked as the #1 neighbour | 80.1% [75.1, 84.6] | 82.4% [72.3, 92.1] |
| Sibling in the top 3 | 92.6% | |
| Mean reciprocal rank | 0.865 | |
| Sibling pair scored >= 0.5 | 59.7% [52.9, 65.9] | 59.3% [45.6, 73.5] |
| Sibling pairs inside the top 1,000 / 4,000 pairs | 45.2% / 67.9% | 46.3% / 68.5% |

Family share of output pairs (a lower bound on precision, since neighbours need not be family): rule
matches 10.6%, model score >= 0.8 11.6%, 0.5 to 0.8 2.5%. Output: 2,936 pairs >= 0.5, 542 households,
607 duplicate-record pairs.

Open questions for the gold set: the true precision, and the score floor for "one side more detailed"
(at K = 500 the rules lower sibling recall from 36.2% to 31.7%, so 0.9 is likely too high).

## What was tried

| Idea | Outcome |
|---|---|
| Train on "same address typed twice" labels | Good at recognising one address (precision 0.92, recall 0.84) but scores neighbours low: it learned that a different house number means a different place |
| Retrain on neighbour labels, with the pipeline's self-match removed | Twins 0.41 to 0.97; the fix that made neighbour scores meaningful |
| Exact-address and same-house rules on top of the model | 633 identical addresses the model alone had scored below 0.5; generic campus addresses had to be excluded first |
| Strict house numbers (not ward or sector numbers) | Halved false same-house links |
| One side more detailed (containment) rule | Main fix for addresses without house numbers, where 90% of missed siblings were |
| Label-bias correction (labels not only from house-number rules) | Small, consistent gain: sibling ranked #1 79.9% to 81.6% |
| Mutual best match | Rejected: +13 to 17 points recall but 3 to 5 times the output, added pairs mostly wrong |
| Pairwise stacker with multiple positives per person | Rejected: higher recall at 0.5 was a calibration effect; no better at a fixed output size |
| Upper-case ids at the database boundary | Fixed 4,541 people whose names never joined (ids differ in case between tables) |
| Remove "S/O", "C/O" phrases from addresses | Required: 6.7% of addresses carried a family name into the model |

## Tests

```sh
python -m pytest
ruff check .
```

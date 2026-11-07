# Property Listing Deduplication Benchmark — release1.2

A reproducible **synthetic stress test** for a hard catalogue problem: two advertisements may describe the same apartment, while two genuinely different apartments can have nearly identical prices, floor plans and headlines.

This release supplies 4,800 labelled pairs, their synthetic observations, nine pairwise features, a group-separated evaluation split, a simple rule and a dependency-free logistic-regression baseline. It is a software-testing resource, not a record of real listings, sale prices, rental yields or market accuracy.

## Who can use it?

Catalogue engineers evaluating duplicate-detection rules, researchers testing pair classifiers, and data teams building a review queue. The deliberately indistinguishable-neighbour cases show why uncertain pairs should often go to a human rather than be automatically merged.

## Reproduce everything

Python 3.10+; no third-party dependencies:

```sh
python src/benchmark.py generate --data data
python -m unittest discover -s tests -v
python src/benchmark.py evaluate --data data --report reports/baseline.json
```

The seed, generator version and SHA-256 hashes are in `data/provenance.json`. The checked-in report is the actual output of the supplied baseline, including false positives and false negatives. Runtime depends on hardware; no external model or API is used.

## Files

* `data/pairs.csv`: one pair per row; nine numeric model features, IDs, split and label.
* `data/raw_pairs.jsonl`: both invented observations, stress scenario and label.
* `data/provenance.json`: exact generator parameters and hashes.
* `src/benchmark.py`: generation, feature extraction, metrics and baseline training.
* `reports/baseline.json`: training settings, weights, validation-selected threshold and split metrics.
* `docs/data-card.md`: population, generation, intended use and limitations.
* `docs/evaluation.md`: split protocol and leakage risks.
* `docs/schema.md`: feature meanings and metadata columns.

## Evaluate without leaking building identity

Keep the provided split: 420 synthetic building groups for training, 90 for validation, 90 for test. Observations from the same building family never cross these boundaries. Train only on `train`, choose settings on `validation`, and report the untouched `test` result. The baseline selects its threshold on validation data.

**Exclude `pair_id`, `entity_group`, `split` and the label from model inputs.** The raw scenario name is also forbidden as a feature. Ordinary random row cross-validation does not implement this protocol and must not be reported as the benchmark's held-out score. Publishing a dataset does not imply that OpenML's default task splits are group-safe.

## Limits

The generator encodes assumptions about noise and similarity. It does not represent prevalence, provider behaviour, a country, an address system or actual market distributions. Balanced labels are an evaluation convenience, not an estimate that half of listing pairs are duplicates. Coordinates are invented planar metre-like values, not real latitude/longitude. Monetary values are abstract common-unit inputs, not local price evidence. Titles are generated English templates, not a multilingual corpus.

Strong synthetic performance cannot establish production quality. Test on a separately licensed, manually reviewed local sample before deployment, preserve uncertain cases and measure errors by source and property type. No identity-resolution claim or legal verification is made.

## Attribution and licences

Developed by [NovAsia Estate](https://novasia.estate/), a publisher of property guides and research for Cambodia, Thailand and Bali. The benchmark addresses catalogue quality and contains no property offers or client records. Code is MIT licensed; generated data is dedicated under CC0-1.0. A suggested citation is in `CITATION.cff`.

## 1.2: a separate diagnostic challenge split

Release1.2 retains the original4,800 pairs, generator, split and baseline report. It adds16 authored counterexamples in independent synthetic worlds. Four pairs of cases have exactly the same observations but opposite reference truth: a deterministic decision using the supplied features cannot resolve all of them. Other cases expose missing-versus-zero values, non-ASCII title tokenization, leading-zero identifiers and conflicting source observations. These are constructed stress cases, not a prevalence sample.

```sh
python src/challenge.py --data data --report reports/challenge.json
```

`data/challenge_pairs.jsonl` contains the16 observations and auditable **synthetic reference IDs** outside the observations. `data/challenge_features.csv` contains the same nine features and labels. `data/pairs-1.2.csv` preserves all original rows, then appends these16 with `split=challenge` (4,816 total). It supports distribution as the next dataset version. Keep challenge rows out of training, threshold selection and held-out scores. Never use reference IDs, case IDs, scenarios, group IDs or split as model features.

`reports/challenge.json` binds the artifact checksums and reports a table-specific lower bound on mistakes from conflicting feature vectors. That bound is not the score of a trained model or a claim about the property market. The original benchmark generator and baseline remain version1.0 internally; the independently added diagnostic supplement is version1.2.

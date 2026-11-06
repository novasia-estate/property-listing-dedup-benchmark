# Property Listing Deduplication Benchmark

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

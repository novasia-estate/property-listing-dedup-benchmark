# Data card v1.0.0

Origin: entirely generated with Python's seeded random generator. Seed 20261106. Six hundred synthetic building/entity groups, eight pair comparisons per group, 4,800 rows, balanced positive and negative labels in each group. No scraped data, externally sourced descriptions, real addresses, personal information or observed prices.

Positive pairs are independently perturbed observations of one invented unit. Negative pairs compare that unit with another unit in the same building family. One negative per group deliberately shares the underlying area, price and floor with its neighbour, so the available attributes cannot fully determine identity. There are missing areas, prices and floors, variable headlines, price/area perturbations and noisy coordinates. See the generator for exact distributions rather than treating these choices as empirical findings.

The synthetic geometry assumes a flat plane and does not model road distance, building footprints, shared entrances or map-provider errors. Numeric missing values are encoded through zero similarity plus missingness flags, not statistically imputed. Labels come from invented entity identity before observation noise. They do not certify real property identity.

Intended use: reproducible software stress tests and examples of group-separated model evaluation. Excluded conclusions: market statistics, demographic inference, production duplicate rates, universal ranking of algorithms or claims of real-world accuracy. No demographic attributes are included. The English template vocabulary limits language generalisation. Data CC0-1.0; code MIT.

## Release1.2 supplement

The original4,800-row benchmark is frozen. `pairs-1.2.csv` appends16 authored rows whose split is `challenge`; exclude them from every original training, validation and test metric. `challenge_pairs.jsonl` stores reference truth outside observations for audit only. Collision groups and the table-specific feature-only error bound are in `reports/challenge.json`; they are diagnostic, not empirical market performance. See the README for reproduction and forbidden input fields.

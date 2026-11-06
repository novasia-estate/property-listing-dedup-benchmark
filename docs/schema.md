# Pair schema

Metadata: `pair_id` is a unique opaque row identifier. `entity_group` groups comparisons from one synthetic building family. `split` is train/validation/test. `is_duplicate` is the binary target (1=same invented unit). Metadata is not a model input.

| Feature | Interpretation |
|---|---|
| title_jaccard | Jaccard similarity of lowercase alphanumeric title tokens |
| area_similarity | 1 minus relative area difference; 0 when either missing |
| price_similarity | 1 minus relative price difference; 0 when either missing |
| location_similarity | exp(-Euclidean distance/100) for synthetic planar coordinates |
| bedrooms_equal | 1 if bedroom counts agree |
| floor_equal | 1 if both known and equal |
| area_missing | 1 if either area missing |
| price_missing | 1 if either price missing |
| floor_missing | 1 if either floor missing |

All features lie in [0,1]. Similarities are rounded to eight decimals in the CSV. Raw observations include the synthetic input values. The raw `scenario` describes generation and is forbidden as a predictor. Zero similarity and missingness have different meanings; preserve the flags.

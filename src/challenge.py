"""Authored diagnostic supplement. Never mix these cases into training or held-out scores."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from benchmark import FEATURES, features

VERSION = '1.2.0'


def cases():
    base = dict(title='Bright apartment', area=70, price=100000, x=0, y=0, bedrooms=2, floor=6)
    result = []

    def add(name, duplicate, left=None, right=None):
        pair_id = f'challenge-{len(result)+1:03d}'
        result.append(dict(pair_id=pair_id, entity_group=pair_id, split='challenge',
            scenario=name, left={**base, **(left or {})}, right={**base, **(right or {})},
            reference_truth={'left':pair_id+'/A', 'right':pair_id+('/A' if duplicate else '/B')},
            is_duplicate=int(duplicate), provenance='authored-independent-synthetic-world'))

    # Each pair of cases has exactly the same observations but opposite authored truth.
    for name, left, right in [
        ('complete_observation_collision', {}, {}),
        ('missing_evidence_collision', dict(area=None,price=None,floor=None), dict(area=None,price=None,floor=None)),
        ('partial_title_overlap_collision', {}, dict(title='Apartment view')),
        ('empty_title_collision', dict(title=''), dict(title='')),
    ]:
        add(name, True, left, right)
        add(name, False, left, right)
    add('same_reference_with_large_source_errors', True, {}, dict(price=250000,x=900,y=600))
    add('different_units_nearby_same_attributes', False, {}, dict(title='City residence',x=2))
    add('explicit_zero_not_missing', True, dict(price=0), dict(price=0,floor=None))
    add('missing_price_not_zero', False, dict(price=None), dict(price=0))
    add('non_ascii_titles_outside_generator_domain', True, dict(title='บ้าน'), dict(title='Дом'))
    add('leading_zero_title_identifiers_remain_distinct', False, dict(title='Unit A01'), dict(title='Unit A1'))
    add('renamed_marketing_title', True, dict(title='Dawn residence'), dict(title='River apartment'))
    add('nearby_units_with_missing_floor', False, dict(floor=None), dict(floor=None,x=1))
    return result


def feature_rows(raw):
    return [dict(pair_id=r['pair_id'],entity_group=r['entity_group'],split='challenge',
                 **dict(zip(FEATURES,[round(v,8) for v in features(r['left'],r['right'])])),
                 is_duplicate=r['is_duplicate']) for r in raw]


def collision_report(rows):
    groups=defaultdict(list)
    for row in rows: groups[tuple(float(row[f]) for f in FEATURES)].append(row)
    conflicting=[group for group in groups.values() if len({r['is_duplicate'] for r in group})>1]
    minimum=sum(min(sum(r['is_duplicate']==label for r in group) for label in (0,1)) for group in conflicting)
    return {'feature_count':len(FEATURES),'cases':len(rows),'conflicting_feature_groups':[
        {'case_ids':[r['pair_id'] for r in group],'labels':[r['is_duplicate'] for r in group]} for group in conflicting],
        'minimum_mistakes_for_deterministic_feature_only_decisions':minimum,
        'table_only_accuracy_upper_bound':(len(rows)-minimum)/len(rows),
        'interpretation':'A table-specific indistinguishability bound, not model performance or market accuracy. Reference truth is authored metadata, never an input feature. Review or abstain when evidence cannot distinguish alternatives.'}


def build(data, report):
    data=Path(data);report=Path(report)
    raw=cases();rows=feature_rows(raw)
    with (data/'pairs.csv').open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames;original=list(reader)
    expected=['pair_id','entity_group','split',*FEATURES,'is_duplicate']
    if fields!=expected or any(r['split']=='challenge' for r in original):
        raise ValueError('Use the original benchmark pairs.csv with its documented fields and splits')
    (data/'challenge_pairs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in raw),encoding='utf-8',newline='\n')
    for name, records in [('challenge_features.csv',rows),('pairs-1.2.csv',original+rows)]:
        with (data/name).open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(records)
    result=collision_report(rows)
    result.update(version=VERSION,original_rows=len(original),combined_rows=len(original)+len(rows),
        split_policy='Keep original train/validation/test rows; exclude every challenge row from fitting, threshold selection and held-out metrics.',
        frozen_inputs_normalized_lf_sha256={name:hashlib.sha256((data/name).read_text(encoding='utf-8').replace('\r\n','\n').encode()).hexdigest() for name in ['pairs.csv','raw_pairs.jsonl','provenance.json']},
        artifact_sha256={name:hashlib.sha256((data/name).read_bytes()).hexdigest() for name in ['challenge_pairs.jsonl','challenge_features.csv','pairs-1.2.csv']})
    report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='data');parser.add_argument('--report',default='reports/challenge.json')
    args=parser.parse_args();result=build(args.data,args.report)
    print(f"Diagnostic supplement: {result['cases']} authored cases; {result['combined_rows']} combined rows. Original benchmark files retained.")

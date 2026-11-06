"""Synthetic property-listing pair benchmark. Python standard library only."""
import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

VERSION = '1.0.0'
FEATURES = ['title_jaccard', 'area_similarity', 'price_similarity', 'location_similarity',
            'bedrooms_equal', 'floor_equal', 'area_missing', 'price_missing', 'floor_missing']

def tokens(text):
    return set(re.findall(r'[a-z0-9]+', text.lower()))

def similarity(a, b):
    return 0.0 if a is None or b is None else max(0.0, 1 - abs(a-b) / max(abs(a), abs(b), 1))

def features(a, b):
    ta, tb = tokens(a['title']), tokens(b['title'])
    distance = math.hypot(a['x']-b['x'], a['y']-b['y'])
    return [len(ta & tb)/len(ta | tb) if ta | tb else 0.0,
            similarity(a['area'], b['area']), similarity(a['price'], b['price']),
            math.exp(-distance / 100), float(a['bedrooms']==b['bedrooms']),
            float(a['floor'] is not None and b['floor'] is not None and a['floor']==b['floor']),
            float(a['area'] is None or b['area'] is None),
            float(a['price'] is None or b['price'] is None),
            float(a['floor'] is None or b['floor'] is None)]

def observation(unit, rng):
    record = dict(unit)
    record.pop('entity')
    record['area'] = round(unit['area'] * rng.uniform(.94, 1.06), 1) if rng.random() > .08 else None
    record['price'] = round(unit['price'] * rng.uniform(.83, 1.17)) if rng.random() > .06 else None
    record['floor'] = unit['floor'] if rng.random() > .12 else None
    record['x'] += rng.gauss(0, 45)
    record['y'] += rng.gauss(0, 45)
    # Copy-like agent headlines can describe different units in the same building.
    templates = ['bright apartment {beds} bedrooms', 'residence {beds} bed city view',
                 'spacious home {beds} bedroom', 'modern apartment {beds} bedrooms']
    record['title'] = rng.choice(templates).format(beds=unit['bedrooms'])
    return record

def generate(seed=20261106, groups=600):
    rng = random.Random(seed)
    group_ids = list(range(groups))
    rng.shuffle(group_ids)
    split_for = {g: ('train' if i < int(groups*.7) else 'validation' if i < int(groups*.85) else 'test') for i,g in enumerate(group_ids)}
    rows, raw = [], []
    for group in range(groups):
        beds = rng.choice([1,2,3])
        area = rng.uniform(25,65) + beds*15
        price = rng.uniform(65000,150000) + area*rng.uniform(800,2500)
        base = dict(entity=f'G{group}-U0', bedrooms=beds, area=area, price=price,
                    floor=rng.randint(2,30), x=group*500.0, y=rng.uniform(-500,500))
        for i in range(8):
            duplicate = i % 2 == 0
            other = dict(base)
            scenario = 'same_unit' if duplicate else 'nearby_similar_unit'
            if not duplicate:
                other.update(entity=f'G{group}-U{i+1}', area=base['area']*rng.uniform(.8,1.22),
                             price=base['price']*rng.uniform(.75,1.3),
                             floor=max(1,base['floor']+rng.choice([-2,-1,0,1,2])))
                # Deliberately hard negatives: identical structural attributes are possible.
                if i == 7:
                    other.update(area=base['area'], price=base['price'], floor=base['floor'])
                    scenario = 'indistinguishable_neighbor'
            left, right = observation(base,rng), observation(other,rng)
            pair_id=f'pair-{group:04d}-{i}'
            row=dict(pair_id=pair_id, entity_group=f'group-{group:04d}', split=split_for[group])
            row.update(zip(FEATURES, [round(x,8) for x in features(left,right)]))
            row['is_duplicate']=int(duplicate)
            rows.append(row)
            raw.append(dict(pair_id=pair_id, entity_group=row['entity_group'], split=row['split'],
                            scenario=scenario,left=left,right=right,is_duplicate=int(duplicate)))
    return rows, raw

def metrics(labels, scores, threshold=.5):
    c=Counter(('tp' if y else 'fp') if s>=threshold else ('fn' if y else 'tn') for y,s in zip(labels,scores))
    precision=c['tp']/max(1,c['tp']+c['fp'])
    recall=c['tp']/max(1,c['tp']+c['fn'])
    return dict(confusion={k:c[k] for k in ['tp','fp','tn','fn']}, precision=precision,
                recall=recall,f1=2*precision*recall/max(1e-12,precision+recall),
                accuracy=(c['tp']+c['tn'])/max(1,len(labels)),count=len(labels))

def sigmoid(x):
    return 1/(1+math.exp(-max(-40,min(40,x))))

def fit(rows, epochs=400):
    weights=[0.0]*(len(FEATURES)+1)
    for _ in range(epochs):
        gradient=[0.0]*len(weights)
        for row in rows:
            x=[1.0]+[float(row[k]) for k in FEATURES]
            error=sigmoid(sum(w*v for w,v in zip(weights,x)))-int(row['is_duplicate'])
            for j,v in enumerate(x):gradient[j]+=error*v
        for j in range(len(weights)):
            weights[j]-=.7*(gradient[j]/len(rows)+(.002*weights[j] if j else 0))
    return weights

def predict(weights,row):
    return sigmoid(weights[0]+sum(w*float(row[k]) for w,k in zip(weights[1:],FEATURES)))

def evaluate(rows):
    splits={s:[r for r in rows if r['split']==s] for s in ['train','validation','test']}
    weights=fit(splits['train'])
    validation=splits['validation']
    thresholds=[i/100 for i in range(20,81)]
    # Only the validation set selects the threshold. Test remains held out.
    threshold=max(thresholds,key=lambda t:metrics([int(r['is_duplicate']) for r in validation],[predict(weights,r) for r in validation],t)['f1'])
    result=dict(version=VERSION,kind='synthetic_stress_test_not_market_accuracy',
                training='batch logistic regression, 400 epochs, lr .7, L2 .002 except intercept',
                feature_order=FEATURES,weights=weights,validation_selected_threshold=threshold,splits={})
    for split,data in splits.items():
        y=[int(r['is_duplicate']) for r in data]
        rule=[float(float(r['area_similarity'])>.95 and float(r['price_similarity'])>.9 and float(r['location_similarity'])>.3 and float(r['bedrooms_equal'])==1) for r in data]
        result['splits'][split]=dict(logistic=metrics(y,[predict(weights,r) for r in data],threshold),
                                     simple_rule=metrics(y,rule),always_duplicate=metrics(y,[1]*len(data)))
    return result

def write_dataset(output,seed=20261106,groups=600):
    output.mkdir(parents=True,exist_ok=True)
    rows,raw=generate(seed,groups)
    with (output/'pairs.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'raw_pairs.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in raw),encoding='utf-8',newline='\n')
    provenance=dict(version=VERSION,seed=seed,groups=groups,rows=len(rows),
                    origin='Entirely synthetic. No scraped listings, observed prices, real coordinates, personal records or property offers.',
                    split='Disjoint building/entity groups: 70% train, 15% validation, 15% test. Preserve these splits.',
                    license='CC0-1.0',files={})
    for name in ['pairs.csv','raw_pairs.jsonl']:
        provenance['files'][name]=hashlib.sha256((output/name).read_bytes()).hexdigest()
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8',newline='\n')
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['generate','evaluate'])
    parser.add_argument('--data',type=Path,default=Path('data'))
    parser.add_argument('--report',type=Path,default=Path('reports/baseline.json'))
    args=parser.parse_args()
    if args.command=='generate':
        rows=write_dataset(args.data)
        print(json.dumps({'rows':len(rows),'output':str(args.data)}))
    else:
        with (args.data/'pairs.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(evaluate(rows),indent=2)+'\n',encoding='utf-8',newline='\n')
        print(args.report)
if __name__=='__main__':main()

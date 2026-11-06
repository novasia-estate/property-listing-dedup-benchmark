import importlib.util
import pathlib
import unittest

spec=importlib.util.spec_from_file_location('benchmark',pathlib.Path(__file__).parents[1]/'src/benchmark.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

class BenchmarkTests(unittest.TestCase):
    def test_reproducibility(self):
        self.assertEqual(b.generate(123,20),b.generate(123,20))
        self.assertNotEqual(b.generate(123,20),b.generate(124,20))
    def test_group_separation(self):
        rows,_=b.generate(groups=40)
        sets={s:{r['entity_group'] for r in rows if r['split']==s} for s in ['train','validation','test']}
        self.assertFalse(sets['train'] & sets['validation'])
        self.assertFalse(sets['train'] & sets['test'])
        self.assertFalse(sets['validation'] & sets['test'])
    def test_balanced_classes_all_splits(self):
        rows,_=b.generate(groups=40)
        for split in ['train','validation','test']:
            labels=[r['is_duplicate'] for r in rows if r['split']==split]
            self.assertEqual(sum(labels)*2,len(labels))
    def test_finite_bounded_features_and_unique_ids(self):
        rows,_=b.generate(groups=20)
        self.assertEqual(len(rows),len({r['pair_id'] for r in rows}))
        for row in rows:
            for k in b.FEATURES:self.assertTrue(0<=row[k]<=1)
    def test_ids_and_split_are_not_model_features(self):
        self.assertFalse({'pair_id','entity_group','split','is_duplicate'} & set(b.FEATURES))
    def test_no_hidden_label_in_raw_observations(self):
        _,raw=b.generate(groups=20)
        for r in raw:
            self.assertNotIn('entity',r['left']);self.assertNotIn('entity',r['right'])
    def test_metrics_with_no_predicted_positives(self):
        r=b.metrics([0,1],[0,0])
        self.assertEqual(r['f1'],0);self.assertEqual(r['confusion'],dict(tp=0,fp=0,tn=1,fn=1))
    def test_identical_values_are_not_always_same_property(self):
        _,raw=b.generate(groups=20)
        hard=[r for r in raw if r['scenario']=='indistinguishable_neighbor']
        self.assertEqual(len(hard),20)
        self.assertTrue(all(r['is_duplicate']==0 for r in hard))
    def test_duplicate_feature_comparison(self):
        a=dict(title='A home',area=50,price=100000,x=0,y=0,bedrooms=1,floor=4)
        self.assertEqual(b.features(a,a)[:6],[1,1,1,1,1,1])
    def test_fit_learns_synthetic_signal(self):
        rows,_=b.generate(groups=30)
        weights=b.fit([r for r in rows if r['split']=='train'],epochs=40)
        self.assertEqual(len(weights),len(b.FEATURES)+1)
        self.assertTrue(all(-100<w<100 for w in weights))

if __name__=='__main__':unittest.main()

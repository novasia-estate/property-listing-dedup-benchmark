import csv,json,pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).parents[1]/'src'))
import challenge as c
ROOT=pathlib.Path(__file__).parents[1]

class ChallengeTests(unittest.TestCase):
    def test_authored_truth_stays_outside_observations_and_features(self):
        raw=c.cases();self.assertEqual(len(raw),16);self.assertEqual(sum(r['is_duplicate'] for r in raw),8)
        for r in raw:
            self.assertEqual(r['is_duplicate'],int(r['reference_truth']['left']==r['reference_truth']['right']))
            self.assertNotIn('reference_truth',r['left']);self.assertNotIn('is_duplicate',r['right'])
            self.assertEqual(r['split'],'challenge')
    def test_observational_collisions_and_positive_error_bound(self):
        raw=c.cases()
        for i in range(0,8,2):
            self.assertEqual(raw[i]['left'],raw[i+1]['left']);self.assertEqual(raw[i]['right'],raw[i+1]['right']);self.assertNotEqual(raw[i]['is_duplicate'],raw[i+1]['is_duplicate'])
        report=c.collision_report(c.feature_rows(raw));self.assertGreaterEqual(len(report['conflicting_feature_groups']),4);self.assertGreaterEqual(report['minimum_mistakes_for_deterministic_feature_only_decisions'],4);self.assertLessEqual(report['table_only_accuracy_upper_bound'],.75)
    def test_features_are_finite_bounded_and_reproducible(self):
        rows=c.feature_rows(c.cases());self.assertEqual(rows,c.feature_rows(c.cases()))
        self.assertEqual(len({r['pair_id'] for r in rows}),16)
        for r in rows:
            for key in c.FEATURES:self.assertTrue(0<=r[key]<=1)
    def test_original4800_rows_and_splits_are_retained(self):
        def read(name):
            with (ROOT/'data'/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
        original=read('pairs.csv');combined=read('pairs-1.2.csv');self.assertEqual(len(original),4800);self.assertEqual(combined[:4800],original);self.assertEqual(len(combined),4816)
        self.assertTrue(all(r['split']=='challenge' for r in combined[4800:]));self.assertFalse({r['entity_group'] for r in original}&{r['entity_group'] for r in combined[4800:]})
    def test_saved_artifacts_and_collision_report_match(self):
        import hashlib
        report=json.loads((ROOT/'reports/challenge.json').read_text(encoding='utf-8'));expected=c.collision_report(c.feature_rows(c.cases()))
        for key,value in expected.items():self.assertEqual(report[key],value)
        for name,digest in report['artifact_sha256'].items():self.assertEqual(hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest(),digest)

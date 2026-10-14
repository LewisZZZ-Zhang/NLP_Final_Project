import unittest,sys,tempfile,json,math
from pathlib import Path
sys.path[:0]=[str(Path('filter').resolve()),str(Path('scripts').resolve())]
from benchmark_data import *
from model import HeuristicAttentionProvider,SentenceInstance,contains_term
from main import AbsoluteThresholdPolicy,DropPercentPolicy,load_dataset
from retest import metrics,softmax
class CleanupTests(unittest.TestCase):
 def test_noise_types(self):
  for x,y in [(False,False),('False',False),(' true ',True),(0,False),(1,True),(None,None),('unknown',None),(2,None)]:self.assertIs(parse_noise(x),y)
 def test_spans(self):
  self.assertEqual(spans('A  B met C','A B'),[[0,4]])
  self.assertEqual(spans('X met X','X'),[[0,1],[6,7]])
  self.assertEqual(spans('Mayfair','May'),[])
 def test_boundary(self):
  self.assertFalse(contains_term('mayfair allegedly','may'));self.assertFalse(contains_term('allegedly','alleged'));self.assertTrue(contains_term('it may rain','may'));self.assertTrue(contains_term('May met X','may'))
 def test_conversion_identity(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'dev.json';objs=[]
   for text,h,t in [('A met B','A','B'),('A likes B','A','B'),('C met D','C','D'),('B met A','B','A'),('X met X and Y','X','Y'),('A met B','Z','B')]:objs.append(dict(sentText=text,relationMentions=[dict(em1Text=h,em2Text=t,label='None')]))
   p.write_text('\n'.join(json.dumps(o) for o in objs));rs=convert_split(p)
   self.assertEqual(rs[0]['h']['id'],rs[1]['h']['id']);self.assertNotEqual(rs[0]['h']['id'],rs[2]['h']['id']);self.assertEqual(rs[4]['validation_status'],'ambiguous_span');self.assertEqual(rs[5]['validation_status'],'missing_span');self.assertTrue(all(r['is_noise'] is None for r in rs));self.assertEqual(len({r['instance_id'] for r in rs}),6)
   q=Path(d)/'v2.txt';write_jsonl(q,rs);bags,_=load_dataset(q);self.assertEqual(sorted(map(len,bags.values())),[1,1,2])
 def test_singleton_allzero(self):
  self.assertEqual(softmax([0],.7),[1]);self.assertEqual(softmax([0,0],.7),[.5,.5]);self.assertEqual(AbsoluteThresholdPolicy(.1,0).select([0]),[False]);self.assertEqual(AbsoluteThresholdPolicy(.1,1).select([0]),[True]);self.assertEqual(DropPercentPolicy(1,0).select([0,0]),[False,False])
 def test_bias_na(self):
  for a,b in zip(softmax([.2,.5],.7),softmax([.4,.7],.7)):self.assertAlmostEqual(a,b)
  self.assertAlmostEqual(sum(softmax([.2*.25,.5*.25],.7)),1)
 def test_softmax_bound(self):
  lower=1/(1+21*math.exp(1.5/.7));self.assertGreater(lower,.0025);self.assertAlmostEqual(softmax([0]+[1.5]*21,.7)[0],lower)
 def test_metrics(self):
  rows=[dict(relation='R',keep=k,is_noise=n) for k,n in [(False,True),(False,False),(True,True),(True,False),(True,None)]]
  m=metrics(rows);self.assertEqual([m[k] for k in ['TP','FP','FN','TN']],[1]*4);self.assertEqual(m['purity'],.5);self.assertEqual(m['unknown'],1)
  self.assertIsNone(metrics([])['purity']);self.assertIsNone(metrics([dict(relation='R',keep=True,is_noise=False)])['precision'])
 def test_legacy_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'old';p.write_text(json.dumps(dict(text='A',h=dict(id='h_0'))))
   with self.assertRaises(ValueError):load_dataset(p)
if __name__=='__main__':unittest.main()

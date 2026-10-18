import sys,unittest
from pathlib import Path
sys.path[:0]=[str(Path('filter').resolve()),str(Path('scripts').resolve())]
from model import SentenceInstance,HeuristicAttentionProvider
from simple_variants import LocalKeywordProvider
from simple_ablation import decisions,fold_id,fit

def instance(text,h,t):
 return SentenceInstance('/people/person/place_of_birth',text,text.lower(),h,t,(text.index(h),text.index(h)+len(h)),(text.index(t),text.index(t)+len(t)),{})

class SimpleVariantTests(unittest.TestCase):
 def test_unrelated_far_keyword(self):
  s=instance('Alice visited Paris, where Bob was born.','Alice','Paris');a=HeuristicAttentionProvider().score_components(s);b=LocalKeywordProvider().score_components(s)
  self.assertGreater(a['keyword'],b['keyword']);self.assertGreater(a['boosts'],b['boosts']);self.assertNotIn('born',b['lexical_context'])
 def test_local_real_evidence(self):
  s=instance('Alice was born in Paris.','Alice','Paris');a=HeuristicAttentionProvider().score_components(s);b=LocalKeywordProvider().score_components(s)
  self.assertEqual(a['raw_score'],b['raw_score']);self.assertIn('born',b['lexical_context'])
 def test_entity_mask(self):
  s=instance('Born visited Paris.','Born','Paris');b=LocalKeywordProvider().score_components(s)
  self.assertNotIn('born',b['lexical_context']);self.assertEqual(b['boosts'],0)
 def test_reverse_and_overlap(self):
  s=instance('Paris is where Alice was born.','Alice','Paris');self.assertIn('born',LocalKeywordProvider().lexical_context(s))
  s=instance('New York is large.','New York','York');self.assertNotIn('york',LocalKeywordProvider().lexical_context(s))
 def test_nonlexical_unchanged(self):
  s=instance('Alice may visit Paris, while Bob was reportedly born elsewhere.','Alice','Paris')
  a=HeuristicAttentionProvider().score_components(s);b=LocalKeywordProvider().score_components(s)
  for k in ['distance','between','order','bias','uncertainty','clutter','overlap']:self.assertEqual(a[k],b[k])
 def test_dual_and_partition_fallback(self):
  rows=[dict(instance_id=str(i),bag_id=g,relation=r,raw_score=v,is_noise=n,sentence_text_id=str(i)) for i,(g,r,v,n) in enumerate([('a','NA',.05,True),('a','NA',.1,False),('b','R',.6,True),('b','R',.9,False)])]
  out=decisions(rows,{'NA':.07,'non-NA':.8});self.assertEqual([r['keep'] for r in out],[False,True,False,True])
  # Removing the highest-scoring sentence in another fold cannot drop singleton.
  self.assertTrue(decisions(rows[:1],{'NA':1.,'non-NA':1.})[0]['keep'])
  self.assertEqual(fold_id('same text hash'),fold_id('same text hash'))
 def test_fit_no_label_leak(self):
  rows=[dict(instance_id=str(i),bag_id=g,relation=r,raw_score=v,is_noise=n,sentence_text_id=str(i)) for i,(g,r,v,n) in enumerate([('a','NA',.05,True),('a','NA',.1,False),('b','R',.6,True),('b','R',.9,False)])]
  config=fit(rows,'dual')['config'];self.assertGreater(config['non-NA'],config['NA'])
  pred=decisions(rows,config);self.assertEqual(sum(not r['keep'] for r in pred),2)
if __name__=='__main__':unittest.main()

import sys,unittest
from pathlib import Path
sys.path[:0]=[str(Path('filter').resolve()),str(Path('scripts').resolve())]
from bag_rejection import decide,no_lexical_evidence,fit

def row(i,rel='R',raw=.4,keyword=0,boosts=0,bag=None,noise=True):
 return dict(instance_id=str(i),bag_id=bag or str(i),relation=rel,raw_score=raw,
             components=dict(keyword=keyword,boosts=boosts),is_noise=noise)

class BagRejectionTests(unittest.TestCase):
 def test_reject_singleton(self):
  r=[row(0)];self.assertTrue(decide(r,'baseline')[0]['keep'])
  self.assertFalse(decide(r,'guarded_rejection',.5)[0]['keep'])
 def test_protect_evidence_and_high_score(self):
  rs=[row(0,keyword=.1),row(1,boosts=.2),row(2,raw=.5)]
  self.assertTrue(all(r['keep'] for r in decide(rs,'guarded_rejection',.5)))
 def test_na_unchanged(self):
  rs=[row(0,rel='NA',raw=0,bag='a'),row(1,rel='NA',raw=.1,bag='a'),row(2,rel='NA',raw=0)]
  baseline=[r['keep'] for r in decide(rs,'baseline')]
  for variant in ['no_non_na_fallback','guarded_rejection']:
   self.assertEqual(baseline,[r['keep'] for r in decide(rs,variant,1.5)])
 def test_can_empty_bag(self):
  rs=[row(0,bag='b'),row(1,bag='b')]
  self.assertFalse(any(r['keep'] for r in decide(rs,'guarded_rejection',.5)))
 def test_partial_bag(self):
  rs=[row(0,bag='b'),row(1,bag='b',keyword=.1)]
  self.assertEqual([False,True],[r['keep'] for r in decide(rs,'guarded_rejection',.5)])
 def test_unguarded_can_drop_lexical(self):
  r=[row(0,keyword=.1)];self.assertFalse(decide(r,'no_non_na_fallback',.5)[0]['keep'])
 def test_fit_retention_and_coverage(self):
  rs=[row(0,raw=.1,noise=True),row(1,raw=.8,noise=False)]
  for v in ['no_non_na_fallback','guarded_rejection']:
   cfg=fit(rs,v);pred=decide(rs,v,cfg['threshold']);self.assertEqual([False,True],[r['keep'] for r in pred])
if __name__=='__main__':unittest.main()

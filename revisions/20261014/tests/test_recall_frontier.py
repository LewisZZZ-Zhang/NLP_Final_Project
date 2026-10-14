import sys,unittest
from pathlib import Path
sys.path[:0]=[str(Path('filter').resolve()),str(Path('scripts').resolve())]
try:from recall_frontier import select_threshold,decisions
except ImportError:select_threshold=None
@unittest.skipIf(select_threshold is None,'Requires training environment')
class RecallFrontierTests(unittest.TestCase):
 def rows(self):return [dict(relation='R',is_noise=y) for y in [True,True,False,False]]
 def test_high_recall_protects_clean_when_possible(self):
  rs=self.rows();choice=select_threshold(rs,[.9,.8,.3,.2],'logistic','non-NA',.95)
  self.assertEqual(choice['metrics']['recall'],1);self.assertEqual(choice['metrics']['clean_retention'],1)
 def test_dropall_boundary(self):
  choice=select_threshold(self.rows(),[.5]*4,'logistic','non-NA',.99)
  self.assertEqual(choice['metrics']['recall'],1);self.assertEqual(choice['metrics']['clean_retention'],0);self.assertIsNone(choice['metrics']['purity'])
 def test_raw_direction(self):
  choice=select_threshold(self.rows(),[.1,.2,.8,.9],'raw','non-NA',.95)
  self.assertEqual(choice['metrics']['recall'],1);self.assertEqual(choice['metrics']['clean_retention'],1)
if __name__=='__main__':unittest.main()

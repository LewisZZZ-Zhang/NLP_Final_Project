import unittest,sys,math
from pathlib import Path
sys.path[:0]=[str(Path('filter').resolve()),str(Path('scripts').resolve())]
from logistic_filter import FEATURES,feature_vector,predict_noise,apply_model

def row(na=False):return dict(relation='NA' if na else 'R',components={k:0. for k in FEATURES[:-1]},text='ignored',raw_score=100)
class LogisticInferenceTests(unittest.TestCase):
 def model(self):return dict(features=FEATURES,mean=[0.]*9,scale=[1.]*9,coefficients=[0.]*9,intercept=0.,thresholds={'NA':.6,'non-NA':.4})
 def test_schema_excludes_manual_score(self):
  r=row();self.assertEqual(len(feature_vector(r)),9);self.assertEqual(sum(feature_vector(r)),0);self.assertEqual(feature_vector(row(True))[-1],1)
 def test_score_and_group_threshold(self):
  m=self.model();self.assertEqual(predict_noise([row()],m),[.5]);self.assertEqual([r['keep'] for r in apply_model([row(),row(True)],m)],[False,True])
 def test_stable_sigmoid(self):
  m=self.model();m['intercept']=1000;self.assertEqual(predict_noise([row()],m),[1]);m['intercept']=-1000;self.assertEqual(predict_noise([row()],m),[0])
 def test_scaling_and_equality(self):
  m=self.model();m['mean'][0]=1;m['scale'][0]=2;m['coefficients'][0]=2
  self.assertAlmostEqual(predict_noise([row()],m)[0],1/(1+math.exp(1)))
  m=self.model();m['thresholds']['non-NA']=.5;self.assertFalse(apply_model([row()],m)[0]['keep'])
 def test_invalid_feature(self):
  r=row();r['components']['distance']=float('nan')
  with self.assertRaises(ValueError):feature_vector(r)

class LogisticTrainingTests(unittest.TestCase):
 def test_export_matches_pipeline_and_scaler_training_only(self):
  try:
   import numpy as np
   from logistic_experiment import fit_pipeline,export_model,predict
  except ImportError:
   self.skipTest('Training tests require the isolated scikit-learn environment')
  rows=[]
  for i in range(30):
   r=row(i%3==0);r['components']['distance']=i/29;r['is_noise']=i>14;rows.append(r)
  pipe=fit_pipeline(rows,.1);model=export_model(pipe,.1,{'NA':.5,'non-NA':.5})
  held=row();held['components']['distance']=1000
  before=pipe.named_steps['standardscaler'].mean_.copy()
  self.assertTrue(np.allclose(predict(pipe,rows+[held]),predict_noise(rows+[held],model),atol=1e-12))
  self.assertTrue(np.array_equal(before,pipe.named_steps['standardscaler'].mean_))
  self.assertAlmostEqual(model['mean'][0],.5)

if __name__=='__main__':unittest.main()

"""Small supervised filter over existing heuristic components.

JSON model export supports inference without scikit-learn or pickle.
"""
import math
FEATURES=['distance','between','keyword','order','boosts','uncertainty','clutter','overlap','is_NA']


def feature_vector(row):
    values=[float(row['components'][k]) for k in FEATURES[:-1]]+[float(row['relation']=='NA')]
    if not all(math.isfinite(v) for v in values):raise ValueError('Non-finite feature')
    return values


def predict_noise(rows,model):
    if model['features']!=FEATURES:raise ValueError('Feature schema mismatch')
    result=[]
    for row in rows:
        x=feature_vector(row)
        z=model['intercept']+sum(w*(v-m)/s for v,m,s,w in zip(x,model['mean'],model['scale'],model['coefficients']))
        p=1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
        result.append(p)
    return result


def apply_model(rows,model):
    out=[]
    for row,p in zip(rows,predict_noise(rows,model)):
        group='NA' if row['relation']=='NA' else 'non-NA';threshold=model['thresholds'][group]
        out.append(dict(row,noise_score=p,threshold=threshold,keep=p<threshold,features=feature_vector(row)))
    return out


def main():
    import argparse,json,sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from retest import load_dataset,score
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    model=json.loads(a.model.read_text());rows=score(load_dataset(a.input)[0]);pred=apply_model(rows,model)
    # Keep every scored decision, not only the retained rows.
    with a.output.open('x') as f:
        for r in pred:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(f'{len(pred)} decisions; {sum(r["keep"] for r in pred)} retained')
if __name__=='__main__':main()

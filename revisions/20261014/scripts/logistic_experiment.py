"""Nested grouped CV. Outer validation and diagnostic test labels never tune."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ.setdefault(key,'1')
import argparse,hashlib,json,sys,time,warnings,platform
from pathlib import Path
import numpy as np
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
from simple_ablation import ROOT,fold_id,write_json,write_rows
from retest import score,load_dataset,metrics
from bag_rejection import decide as rule_decide,fit as rule_fit,summary
from logistic_filter import FEATURES,feature_vector,predict_noise,apply_model

CS=[.01,.1,1.,10.]
THRESHOLDS=[i/100 for i in range(101)]+[1.000001]


def fit_pipeline(rows,C):
    X=np.array([feature_vector(r) for r in rows]);y=np.array([int(r['is_noise']) for r in rows])
    if any(r['is_noise'] is None for r in rows):raise ValueError('Unknown labels cannot train')
    pipeline=make_pipeline(StandardScaler(),LogisticRegression(C=C,penalty='l2',solver='lbfgs',max_iter=2000,tol=1e-8,class_weight=None))
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning);pipeline.fit(X,y)
    return pipeline


def export_model(pipe,C,thresholds):
    scaler=pipe.named_steps['standardscaler'];clf=pipe.named_steps['logisticregression']
    assert list(clf.classes_)==[0,1]
    return dict(features=FEATURES,mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coefficients=clf.coef_[0].tolist(),intercept=float(clf.intercept_[0]),thresholds=thresholds,C=C,classes=[False,True],positive_class='noise',iterations=clf.n_iter_.tolist(),sklearn_version=sklearn.__version__,interpretation='uncalibrated model probability estimate, not a verified clean/noise probability')


def predict(pipe,rows):
    return pipe.predict_proba(np.array([feature_vector(r) for r in rows]))[:,1]


def pick_threshold(rows,probs,group):
    chosen=None;curve=[]
    subset=[(r,p) for r,p in zip(rows,probs) if (r['relation']=='NA')==(group=='NA')]
    for t in THRESHOLDS:
        pred=[dict(r,keep=float(p)<t) for r,p in subset];m=metrics(pred)
        feasible=m['clean_retention'] is not None and m['clean_retention']>=.95
        curve.append(dict(threshold=t,metrics=m,feasible=feasible))
        if feasible:
            key=(m['recall'] if m['recall'] is not None else -1,m['purity'],m['clean_retention'],t)
            if chosen is None or key>chosen[0]:chosen=(key,t,m)
    if chosen is None:raise ValueError('No feasible threshold')
    return dict(threshold=chosen[1],metrics=chosen[2],curve=curve)


def tune(rows,out,label):
    """Inner folds are the remaining outer group labels; five for full dev."""
    folds=sorted({fold_id(r['sentence_text_id']) for r in rows});candidates=[];best=None
    for C in CS:
        probs=np.full(len(rows),np.nan);lineage=[]
        for f in folds:
            train_indices=[i for i,r in enumerate(rows) if fold_id(r['sentence_text_id'])!=f]
            val_indices=[i for i,r in enumerate(rows) if fold_id(r['sentence_text_id'])==f]
            train=[rows[i] for i in train_indices];val=[rows[i] for i in val_indices]
            assert {r['sentence_text_id'] for r in train}.isdisjoint(r['sentence_text_id'] for r in val)
            pipe=fit_pipeline(train,C);probs[val_indices]=predict(pipe,val)
            model=export_model(pipe,C,{})
            assert np.allclose(model['mean'],np.mean([feature_vector(r) for r in train],axis=0))
            lineage.append(dict(fold=f,train_ids=[r['instance_id'] for r in train],validation_ids=[r['instance_id'] for r in val],model=model))
        assert np.isfinite(probs).all()
        selected={g:pick_threshold(rows,probs,g) for g in ['NA','non-NA']}
        loss=float(log_loss([r['is_noise'] for r in rows],probs,labels=[False,True]))
        m=selected['non-NA']['metrics'];key=(m['recall'],m['purity'],-loss,-C)
        candidate=dict(C=C,thresholds={g:v['threshold'] for g,v in selected.items()},selected=selected,inner_log_loss=loss)
        candidates.append(candidate)
        write_json(out/f'{label}_C{C}_lineage.json',lineage)
        write_rows(out/f'{label}_C{C}_inner_predictions.jsonl',[dict(instance_id=r['instance_id'],sentence_text_id=r['sentence_text_id'],inner_fold=fold_id(r['sentence_text_id']),is_noise=r['is_noise'],noise_score=float(p)) for r,p in zip(rows,probs)])
        if best is None or key>best[0]:best=(key,candidate)
    write_json(out/f'{label}_tuning.json',candidates)
    return best[1]


def prediction_metrics(rows):
    result=summary(rows);extra={}
    for g in ['overall','NA','non-NA']:
        rs=[r for r in rows if r['is_noise'] is not None and (g=='overall' or (r['relation']=='NA')==(g=='NA'))]
        y=[r['is_noise'] for r in rs];p=[r['noise_score'] for r in rs]
        extra[g]=dict(roc_auc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,average_precision=float(average_precision_score(y,p)) if any(y) else None,brier=float(brier_score_loss(y,p)) if y else None)
    result['score_diagnostics']=extra;return result


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    files=['filter/logistic_filter.py','filter/model.py','scripts/logistic_experiment.py','data/arnor_benchmark_v2/arnor_dev.txt','data/arnor_benchmark_v2/arnor_test.txt','data/arnor_benchmark_v2/arnor_test_noise.txt']
    protocol=dict(features=FEATURES,excluded=['raw_score','manual bias','text','entity names','relation one-hot','bag size'],C_grid=CS,threshold_grid=THRESHOLDS,model='StandardScaler + L2 LogisticRegression; lbfgs; unweighted labels; learned unpenalized intercept; class 1 is noise',
                  folds='same 5 normalized-text groups as earlier; outer 5; inner remaining 4; full-dev final tuning inner 5; fit scaler only on each inner training partition; no overlapping sentence groups',
                  objective='for each C choose separate NA/non-NA thresholds maximizing group noise recall subject to group clean retention>=.95; threshold ties purity, clean retention, higher threshold; choose C by non-NA recall, non-NA purity, lower overall inner log loss, smaller C',
                  comparison='rules retain their original NA policy, so report NA and non-NA separately; guard retuned on outer training only; LR imposes stronger NA clean-retention constraint and no bag minimum or relation-coverage constraint',
                  adoption='research candidate only if pooled outer non-NA clean retention>=.95, recall strictly greater than guarded comparator and non-NA purity no lower; report relation erasure and every fold; test not used for selection; defaults unchanged',
                  caveats='dev/test already inspected; nested CV cannot undo prior design influence. No pristine holdout; score probabilities not calibrated. Original train has no labels and is not used.',
                  versions=dict(python=sys.version,numpy=np.__version__,sklearn=sklearn.__version__,platform=platform.platform()),sources={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},references=['https://scikit-learn.org/stable/modules/compose.html','https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html'])
    write_json(out/'protocol.json',protocol)
    start=time.perf_counter();rows=score(load_dataset(ROOT/'data/arnor_benchmark_v2/arnor_dev.txt')[0]);assert all(r['is_noise'] is not None for r in rows)
    write_json(out/'sentence_folds.json',{r['sentence_text_id']:fold_id(r['sentence_text_id']) for r in rows})
    oof=[];rule_oof=[];guard_oof=[];outer=[]
    for f in range(5):
        train=[r for r in rows if fold_id(r['sentence_text_id'])!=f];val=[r for r in rows if fold_id(r['sentence_text_id'])==f]
        selected=tune(train,out,f'outer{f}');pipe=fit_pipeline(train,selected['C']);model=export_model(pipe,selected['C'],selected['thresholds']);pred=apply_model(val,model)
        assert np.allclose(predict(pipe,val),[r['noise_score'] for r in pred],atol=1e-12)
        # Rebuild bag sizes inside outer validation for stratified diagnostics.
        sizes={r['instance_id']:r['bag_size'] for r in rule_decide(val,'baseline')}
        pred=[dict(r,fold=f,bag_size=sizes[r['instance_id']]) for r in pred];oof+=pred
        rule_oof += [dict(r,fold=f) for r in rule_decide(val,'baseline')]
        guard_cfg=rule_fit(train,'guarded_rejection');guard_pred=[dict(r,fold=f) for r in rule_decide(val,'guarded_rejection',guard_cfg['threshold'])];guard_oof+=guard_pred
        fold=dict(fold=f,train_ids=[r['instance_id'] for r in train],validation_ids=[r['instance_id'] for r in val],model=model,validation=prediction_metrics(pred),guard_threshold=guard_cfg['threshold'],guard_validation=summary(guard_pred));outer.append(fold)
        write_json(out/f'outer{f}_model.json',fold)
        print('outer',f,'C',selected['C'],'thresholds',selected['thresholds'],'non-NA',fold['validation']['non-NA'],flush=True)
    assert len(oof)==len(rows) and len({r['instance_id'] for r in oof})==len(rows)
    results=dict(logistic={'oof_dev':prediction_metrics(oof)},baseline={'oof_dev':summary(rule_oof)},guarded={'oof_dev':summary(guard_oof)})
    for n,rs in [('logistic',oof),('baseline',rule_oof),('guarded',guard_oof)]:write_rows(out/f'oof_{n}.jsonl',rs)
    selected=tune(rows,out,'final');t=time.perf_counter();pipe=fit_pipeline(rows,selected['C']);fit_seconds=time.perf_counter()-t;model=export_model(pipe,selected['C'],selected['thresholds'])
    write_json(out/'model.json',model);guard_cfg=rule_fit(rows,'guarded_rejection');write_json(out/'guard_config.json',dict(threshold=guard_cfg['threshold']))
    n=results['logistic']['oof_dev']['non-NA'];g=results['guarded']['oof_dev']['non-NA'];passed=n['clean_retention']>=.95 and n['recall']>g['recall'] and n['purity']>=g['purity']
    write_json(out/'dev_decision.json',dict(candidate_passed=passed,reason='Predeclared non-NA OOF gate; diagnostics not yet evaluated; no automatic replacement'))
    seen={r['sentence_text_id'] for r in rows}
    for split in ['test','test_noise']:
        test=score(load_dataset(ROOT/f'data/arnor_benchmark_v2/arnor_{split}.txt')[0]);pred=apply_model(test,model)
        assert np.allclose(predict(pipe,test),[r['noise_score'] for r in pred],atol=1e-12)
        results['logistic'][split]=prediction_metrics(pred);write_rows(out/f'{split}_logistic.jsonl',pred)
        results['logistic'][split+'_no_dev_text']=prediction_metrics(apply_model([r for r in test if r['sentence_text_id'] not in seen],model))
        for name,variant in [('baseline','baseline'),('guarded','guarded_rejection')]:
            rp=rule_decide(test,variant,guard_cfg['threshold']);results[name][split]=summary(rp);write_rows(out/f'{split}_{name}.jsonl',rp)
    times=[]
    for i in range(7):
        t=time.perf_counter();predict_noise(rows,model);elapsed=time.perf_counter()-t
        if i>=2:times.append(elapsed)
    write_json(out/'timing.json',dict(full_experiment_seconds=time.perf_counter()-start,final_fit_seconds=fit_seconds,inference_4241_feature_rows_seconds=times,scope='inference excludes feature extraction, reading, writing; single thread requested'))
    write_json(out/'results.json',results);write_json(out/'outer_summary.json',outer)
    print('Complete, candidate passed:',passed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'results/logistic_20261014');a=p.parse_args();run(a.out)

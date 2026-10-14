"""Predeclared conservative bag-rejection experiment. Standard library only."""
import argparse,collections,hashlib,json,platform,sys
from pathlib import Path
from simple_ablation import ROOT,fold_id,write_json,write_rows,decisions as baseline_decisions
from retest import score,metrics,softmax,load_dataset

VARIANTS=['baseline','no_non_na_fallback','guarded_rejection']
GRID=[i/100 for i in range(151)]


def no_lexical_evidence(row):
    # Includes both the existing full-text and between keyword inventories.
    return row['components']['keyword']==0 and row['components']['boosts']==0


def decide(rows,variant,threshold=.07):
    if variant not in VARIANTS:raise ValueError(variant)
    base=baseline_decisions(rows,{'NA':.07,'non-NA':.07})
    out=[]
    for r in base:
        keep=r['keep'];reason='baseline'
        if r['relation']!='NA':
            if variant=='no_non_na_fallback':
                keep=r['raw_score']>=threshold
                reason='raw_threshold_without_fallback'
            elif variant=='guarded_rejection' and no_lexical_evidence(r) and r['raw_score']<threshold:
                keep=False;reason='no_lexical_evidence_and_low_raw'
        out.append(dict(r,baseline_keep=r['keep'],keep=keep,rejection_threshold=threshold,
                        no_lexical_evidence=no_lexical_evidence(r),decision_reason=reason))
    return out


def summary(rows):
    groups={'overall':rows,'NA':[r for r in rows if r['relation']=='NA'],
            'non-NA':[r for r in rows if r['relation']!='NA']}
    for label,subset in list(groups.items()):
        if label!='overall':
            groups[label+'_singleton']=[r for r in subset if r['bag_size']==1]
            groups[label+'_multi']=[r for r in subset if r['bag_size']>1]
    result={k:metrics(v) for k,v in groups.items()}
    bags=collections.defaultdict(list)
    for r in rows:bags[(r.get('fold'),r['bag_id'])].append(r)
    result['bag_coverage']=dict(total=len(bags),emptied=sum(not any(r['keep'] for r in bag) for bag in bags.values()))
    result['per_relation']={rel:metrics([r for r in rows if r['relation']==rel]) for rel in sorted({r['relation'] for r in rows})}
    return result


def fit(rows,variant):
    if variant=='baseline':return dict(threshold=.07,curve=[])
    curve=[];best=None
    for t in GRID:
        pred=decide(rows,variant,t);m=metrics([r for r in pred if r['relation']!='NA'])
        valid=m['clean_retention'] is not None and m['clean_retention']>=.95 and not m['emptied_relations']
        curve.append(dict(threshold=t,metrics=m,feasible=valid))
        if valid:
            key=(m['recall'] if m['recall'] is not None else -1,m['purity'],m['clean_retention'],-t)
            if best is None or key>best[0]:best=(key,t)
    if best is None:raise ValueError('No feasible threshold')
    return dict(threshold=best[1],curve=curve)


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    files=['filter/model.py','scripts/bag_rejection.py','data/arnor_benchmark_v2/arnor_dev.txt',
           'data/arnor_benchmark_v2/arnor_test.txt','data/arnor_benchmark_v2/arnor_test_noise.txt']
    protocol=dict(variants=VARIANTS,grid=GRID,baseline='original repaired global scorer; raw>=.07 and min_keep=1; NA unchanged in all variants',
                  guarded='start from baseline; for non-NA only, override keep to drop iff keyword==0 AND boosts==0 AND raw<t; no restoring last instance',
                  lexical_definition='use existing full-text AND between keyword score, including generic prepositions/punctuation; no local-window change',
                  unguarded='non-NA keep iff raw>=t, with min_keep=0; tune t so comparison is not a useless .07-only test',
                  fit_objective='maximize non-NA noise recall subject to non-NA clean retention>=.95 and no non-NA relation emptied; ties: non-NA purity, clean retention, lower threshold',
                  folds='reuse previous deterministic five normalized-text folds; fit on 4, evaluate 1; rebuild bags within each partition',
                  adoption='OOF non-NA clean retention>=.95, recall strictly greater than baseline, non-NA and overall purity at least baseline, and no relation emptied; if both pass prefer recall, then purity; otherwise retain baseline',
                  limitations='dev repeatedly inspected; OOF exploratory; old test/test_noise are diagnostic only; valid spans only; no guarantee a fit constraint generalizes',
                  environment=dict(python=sys.version,platform=platform.platform()),source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
    write_json(out/'protocol.json',protocol)
    rows=score(load_dataset(ROOT/'data/arnor_benchmark_v2/arnor_dev.txt')[0])
    assignments={r['sentence_text_id']:fold_id(r['sentence_text_id']) for r in rows}
    write_json(out/'sentence_folds.json',assignments)
    frozen={};results={}
    for name in VARIANTS:
        oof=[];folds=[]
        for f in range(5):
            train=[r for r in rows if assignments[r['sentence_text_id']]!=f]
            val=[r for r in rows if assignments[r['sentence_text_id']]==f]
            assert {r['sentence_text_id'] for r in train}.isdisjoint(r['sentence_text_id'] for r in val)
            cfg=fit(train,name);pred=[dict(r,fold=f) for r in decide(val,name,cfg['threshold'])]
            oof+=pred;folds.append(dict(fold=f,fit=cfg,validation=summary(pred)))
            print(name,f,cfg['threshold'],flush=True)
        assert len(oof)==len(rows) and len({r['instance_id'] for r in oof})==len(rows)
        write_rows(out/f'oof_{name}.jsonl',oof);write_json(out/f'cv_{name}.json',folds)
        cfg=fit(rows,name);frozen[name]=cfg
        results[name]=dict(oof_dev=summary(oof),fit_dev=summary(decide(rows,name,cfg['threshold'])))
    base=results['baseline']['oof_dev'];eligible=[]
    for name in VARIANTS[1:]:
        m=results[name]['oof_dev'];n=m['non-NA']
        if (n['clean_retention']>=.95 and n['recall']>base['non-NA']['recall'] and
            n['purity']>=base['non-NA']['purity'] and m['overall']['purity']>=base['overall']['purity'] and not n['emptied_relations']):eligible.append(name)
    chosen=max(eligible,key=lambda n:(results[n]['oof_dev']['non-NA']['recall'],results[n]['oof_dev']['non-NA']['purity'])) if eligible else 'baseline'
    write_json(out/'frozen_configs.json',frozen);write_json(out/'dev_decision.json',dict(chosen=chosen,eligible=eligible,note='Frozen before computing diagnostic test results. No automatic default change.'))
    for split in ['test','test_noise']:
        test=score(load_dataset(ROOT/f'data/arnor_benchmark_v2/arnor_{split}.txt')[0])
        for name in VARIANTS:
            pred=decide(test,name,frozen[name]['threshold']);write_rows(out/f'{split}_{name}.jsonl',pred);results[name][split]=summary(pred)
            subset=[r for r in test if r['sentence_text_id'] not in assignments]
            results[name][split+'_no_dev_text']=summary(decide(subset,name,frozen[name]['threshold']))
    write_json(out/'results.json',results)
    print('Complete; dev selection:',chosen,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'results/bag_rejection_20261014');a=p.parse_args();run(a.out)

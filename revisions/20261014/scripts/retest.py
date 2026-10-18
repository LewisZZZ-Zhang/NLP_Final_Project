"""Reproducible CPU audit. Run from repository root; only stdlib required."""
import sys,json,math,random,time,statistics,platform,subprocess,collections,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'filter'))
from model import HeuristicAttentionProvider, contains_term
from main import load_dataset,AbsoluteThresholdPolicy,TopKPolicy,RelativeMarginPolicy,DropPercentPolicy
from benchmark_data import read_objects
OUT=ROOT/'results/cleanup_20261014'; OUT.mkdir(parents=True,exist_ok=True)
def save(n,x): (OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def metrics(rows):
    c=collections.Counter(); rel=collections.Counter(); keptrel=collections.Counter()
    for r in rows:
        rel[r['relation']]+=1
        if r['keep']: keptrel[r['relation']]+=1
        if r['is_noise'] is None: c['unknown']+=1; continue
        c[('FN' if r['keep'] else 'TP') if r['is_noise'] else ('TN' if r['keep'] else 'FP')]+=1
    tp,fp,fn,tn=[c[k] for k in ['TP','FP','FN','TN']]
    ratio=lambda a,b:a/b if b else None
    return dict(c,TP=tp,FP=fp,FN=fn,TN=tn,N=len(rows),precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn),noise_f1=ratio(2*tp,2*tp+fp+fn),purity=ratio(tn,tn+fn),clean_retention=ratio(tn,tn+fp),deletion_rate=ratio(sum(not r['keep'] for r in rows),len(rows)),emptied_relations=sorted(set(rel)-set(keptrel)))
def softmax(vals,T):
    m=max(vals); ex=[math.exp((v-m)/T) for v in vals]; z=sum(ex); return [v/z for v in ex]
def score(bags,provider=None):
    p=provider or HeuristicAttentionProvider(); rows=[]
    for key,bag in bags.items():
        for s in bag:
            comp=p.score_components(s)
            rows.append(dict(s.data,bag_id=key,bag_size=len(bag),components=comp,raw_score=comp['raw_score']))
    return rows

def decide(rows,cfg):
    result=[]; grouped=collections.defaultdict(list)
    for r in rows: grouped[r['bag_id']].append(r)
    kind=cfg['policy']; v=cfg.get('value',0); minimum=cfg.get('min_keep',1)
    for bag in grouped.values():
        weights=softmax([r['raw_score'] for r in bag],cfg.get('temperature',.7))
        if kind in ('raw','absolute'): mask=AbsoluteThresholdPolicy(v,minimum).select([r['raw_score'] for r in bag] if kind=='raw' else weights)
        elif kind=='topk': mask=TopKPolicy(v,minimum).select(weights)
        elif kind=='relative': mask=RelativeMarginPolicy(v,minimum).select(weights)
        elif kind=='drop_pct': mask=DropPercentPolicy(v,minimum).select(weights)
        else: mask=[kind=='keep_all']*len(bag)
        result.extend(dict(r,keep=k,bag_weight=w) for r,k,w in zip(bag,mask,weights))
    return result

def dump_rows(name,rows):
    with (OUT/(name+'.jsonl')).open('w') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')

def strata(rows):
    result={}
    for dim,fn in [('relation',lambda r:r['relation']),('NA',lambda r:'NA' if r['relation']=='NA' else 'non-NA'),('bag_size',lambda r:'1' if r['bag_size']==1 else '2-5' if r['bag_size']<=5 else '6-10' if r['bag_size']<=10 else '11+')]:
        groups=collections.defaultdict(list)
        for r in rows:groups[fn(r)].append(r)
        result[dim]={k:metrics(v) for k,v in groups.items()}
    return result

class Ablation(HeuristicAttentionProvider):
    def __init__(self,mode): super().__init__(); self.mode=mode
    def score_components(self,s):
        c=super().score_components(s); weights=[.28,.18,.38,.16]
        if self.mode=='initial_weights':weights=[.35,.25,.30,.10]
        if self.mode in ('initial_weights','four_rules'):
            for k in ('uncertainty','clutter','overlap'):c[k]=0
        elif self.mode.startswith('no_'):c[self.mode[3:]]=0
        raw=sum(w*c[k] for w,k in zip(weights,['distance','between','keyword','order']))+c['bias']+c['boosts']-sum(c[k] for k in ['uncertainty','clutter','overlap'])
        c['raw_score']=max(0,min(1.5,raw*(.25 if s.relation=='NA' else 1)));return c

def run():
    # Protocol is persisted before dev scoring or parameter search.
    configs=[]
    for minimum in [0,1]:
        configs += [dict(policy='raw',value=i/100,min_keep=minimum,temperature=.7) for i in range(151)]
        for T in [.3,.7,1.2]:
            for kind,vs in [('topk',[1,2,3,5]),('relative',[0,.01,.05,.1,.2,.5,1]),('drop_pct',[0,.1,.25,.5,.75,1])]:
                configs += [dict(policy=kind,value=v,min_keep=minimum,temperature=T) for v in vs]
    save('protocol.json',dict(objective='maximize dev retained purity; clean retention >= .80; at least one retained instance per observed relation; tie: clean retention then fewer removals then first config',configs=configs,unknown_labels='excluded from quality denominators',invalid_spans='quarantine, excluded from scoring, never auto-clean',duplicates='preserved; deduplicated sensitivity separately',test='previously inspected; not pristine held-out',random_seeds=list(range(20))))
    data={s:read_objects(ROOT/f'data/arnor_benchmark_v2/arnor_{s}.txt') for s in ['train','dev','test','test_noise']}
    audit={}; overlaps={}
    for s,rs in data.items():
        groups=collections.Counter(json.dumps([r['relation'],r['h']['id'],r['t']['id']]) for r in rs)
        audit[s]=dict(instances=len(rs),sentences=len({r['source_sentence_id'] for r in rs}),text_sentences=len({r['sentence_text_id'] for r in rs}),bags=len(groups),bag_size_distribution=dict(collections.Counter(groups.values())),largest_bag=max(groups.values()),relations=dict(collections.Counter(r['relation'] for r in rs)),statuses=dict(collections.Counter(r['validation_status'] for r in rs)),unknown=sum(r['is_noise'] is None for r in rs),duplicates=sum(r['duplicate_of'] is not None for r in rs))
    for a,b in itertools.combinations(data,2):
        sa={r['sentence_text_id'] for r in data[a]}; sb={r['sentence_text_id'] for r in data[b]}
        def keys(s):return {(r['sentence_text_id'],r['relation'],r['h']['id'],r['t']['id']) for r in data[s]}
        overlaps[a+'__'+b]=dict(shared_texts=len(sa&sb),shared_instances=len(keys(a)&keys(b)),introduced_by_conversion=0)
    save('data_audit.json',dict(splits=audit,overlaps=overlaps))
    dump_rows('quarantine',[r for rs in data.values() for r in rs if r['validation_status']!='valid'])
    # Stage 2: new grouping and validated spans, ORIGINAL scorer and strategy.
    import importlib.util
    spec=importlib.util.spec_from_file_location('legacy_model',ROOT/'evidence/legacy/model.py'); legacy=importlib.util.module_from_spec(spec);sys.modules['legacy_model']=legacy;spec.loader.exec_module(legacy)
    bags={s:load_dataset(ROOT/f'data/arnor_benchmark_v2/arnor_{s}.txt')[0] for s in ['dev','test','test_noise']}
    stage2={}
    for s,bs in bags.items():
        rs=score(bs); old=legacy.HeuristicAttentionProvider()
        for r,inst in zip(rs,[x for bag in bs.values() for x in bag]):r['raw_score']=old._score_sentence(inst);r['components']=None
        out=decide(rs,dict(policy='absolute',value=.0025,min_keep=1,temperature=.7));stage2[s]=metrics(out);dump_rows('stage2_'+s,out)
    save('stage2.json',stage2)
    dev=score(bags['dev']); sweep=[]
    for cfg in configs:
        m=metrics(decide(dev,cfg)); sweep.append(dict(config=cfg,metrics=m))
    save('dev_sweep.json',sweep)
    eligible=[x for x in sweep if x['metrics']['clean_retention'] is not None and x['metrics']['clean_retention']>=.8 and not x['metrics']['emptied_relations']]
    best=max(eligible,key=lambda x:(x['metrics']['purity'],x['metrics']['clean_retention'],-x['metrics']['deletion_rate']))
    save('frozen_config.json',best)
    final={}; baselines={}; ablations={}; errors={}; timings={}
    for s,bs in bags.items():
        rs=score(bs); out=decide(rs,best['config']); final[s]=metrics(out);dump_rows('final_'+s,out);save('strata_'+s+'.json',strata(out))
        final[s]['deduplicated']=metrics([r for r in out if r['duplicate_of'] is None])
        if s!='dev':
            seen={r['sentence_text_id'] for r in data['dev']}|{r['sentence_text_id'] for r in data['train']}
            final[s]['no_train_dev_text_overlap']=metrics([r for r in out if r['sentence_text_id'] not in seen])
        baselines[s]={k:metrics(decide(rs,dict(policy=k))) for k in ['keep_all','drop_all']}
        n_drop=sum(not r['keep'] for r in out); randoms=[]
        for seed in range(20):
            dropped=set(random.Random(seed).sample(range(len(rs)),n_drop)); rr=[dict(r,keep=i not in dropped) for i,r in enumerate(rs)];randoms.append(dict(seed=seed,metrics=metrics(rr)))
        baselines[s]['random_matched_removals']=randoms
        ablations[s]={}
        for mode in ['initial_weights','four_rules','no_uncertainty','no_clutter','no_overlap']:
            ar=score(bs,Ablation(mode)); fixed=decide(ar,best['config']); ranked=sorted(range(len(ar)),key=lambda i:ar[i]['raw_score']);dropped=set(ranked[:n_drop]);matched=[dict(r,keep=i not in dropped) for i,r in enumerate(ar)]
            ablations[s][mode]=dict(fixed=metrics(fixed),matched_global_retention=metrics(matched))
            dump_rows('ablation_'+s+'_'+mode,fixed)
        # Also report final scorer globally ranked at same rate for a fair mechanism comparison.
        ranked=sorted(range(len(rs)),key=lambda i:rs[i]['raw_score']); ds=set(ranked[:n_drop]);ablations[s]['final_global_rank_matched']=metrics([dict(r,keep=i not in ds) for i,r in enumerate(rs)])
        errors[s]={}
        for cell in ['TP','FP','FN','TN']:
            subset=[r for r in out if r['is_noise'] is not None and (('FN' if r['keep'] else 'TP') if r['is_noise'] else ('TN' if r['keep'] else 'FP'))==cell]
            errors[s][cell]=random.Random(42).sample(subset,min(8,len(subset)))
        measurements=[]
        for rep in range(9):
            t=time.perf_counter(); fresh=load_dataset(ROOT/f'data/arnor_benchmark_v2/arnor_{s}.txt')[0];t1=time.perf_counter();freshrows=score(fresh);t2=time.perf_counter();selected=decide(freshrows,best['config']);t3=time.perf_counter();dump_rows('timing_output',selected);t4=time.perf_counter();score(fresh);t5=time.perf_counter()
            if rep>=2:measurements.append(dict(read=t1-t,score_cold=t2-t1,select=t3-t2,write=t4-t3,score_warm=t5-t4,total=t4-t))
        timings[s]={k:dict(median=statistics.median(m[k] for m in measurements),min=min(m[k] for m in measurements),max=max(m[k] for m in measurements),stdev=statistics.stdev(m[k] for m in measurements)) for k in measurements[0]}
    save('final_metrics.json',final);save('baselines.json',baselines);save('ablations.json',ablations);save('error_samples.json',errors)
    save('cpu_timing.json',dict(environment=platform.platform(),processor=platform.machine()+'; exact chip unavailable under sandbox',python=sys.version,threads=1,warmups=2,repeats=7,plotting='excluded',timings=timings))
    print(json.dumps(dict(best=best,final=final),indent=2))
if __name__=='__main__':run()

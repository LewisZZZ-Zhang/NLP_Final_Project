"""Sentence-grouped dev CV for a predeclared 2x2 ablation; stdlib only."""
import argparse, collections, hashlib, json, platform, sys, time
from pathlib import Path
from retest import ROOT, score, metrics, softmax, load_dataset
from model import HeuristicAttentionProvider
from simple_variants import LocalKeywordProvider

GRID = [i/100 for i in range(151)]
VARIANTS = {'baseline': ('global', 'single'), 'dual': ('global', 'dual'),
            'local': ('local', 'single'), 'combined': ('local', 'dual')}


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def write_rows(path, rows):
    with path.open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def fold_id(sentence_text_id):
    # Identical normalized text must remain together, even across articles.
    return int(hashlib.sha256(('simple-ablation-v1:' + sentence_text_id).encode()).hexdigest(), 16) % 5


def decisions(rows, config):
    """Rebuild bags only from the passed partition; no cross-fold fallback."""
    bags = collections.defaultdict(list)
    for r in rows:
        bags[r['bag_id']].append(r)
    result = []
    for bag in bags.values():
        threshold = config['NA' if bag[0]['relation'] == 'NA' else 'non-NA']
        raw = [r['raw_score'] for r in bag]
        keep = [s >= threshold for s in raw]
        if not any(keep):
            keep[max(range(len(raw)), key=lambda i: raw[i])] = True
        weights = softmax(raw, .7)
        result.extend(dict(r, keep=k, bag_weight=w, bag_size=len(bag), threshold=threshold)
                      for r, k, w in zip(bag, keep, weights))
    return result


def summarize(rows):
    result = {'overall': metrics(rows)}
    for group in ['NA', 'non-NA']:
        result[group] = metrics([r for r in rows if (r['relation']=='NA') == (group=='NA')])
    return result


def combine(a, b):
    tp, fp, fn, tn = [a[k]+b[k] for k in ['TP','FP','FN','TN']]
    total = a['N']+b['N']
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, N=total,
                purity=tn/(tn+fn) if tn+fn else None,
                clean_retention=tn/(tn+fp) if tn+fp else None,
                deletion_rate=(tp+fp)/total if total else None)


def fit(rows, family):
    """Compute per-group threshold curves, then exhaustively combine counts."""
    curves = {}
    for group in ['NA', 'non-NA']:
        subset = [r for r in rows if (r['relation']=='NA') == (group=='NA')]
        curves[group] = [metrics(decisions(subset, {'NA':t, 'non-NA':t})) for t in GRID]
    pairs = ((i,j) for i in range(len(GRID)) for j in range(len(GRID))) if family=='dual' else ((i,i) for i in range(len(GRID)))
    best = None; feasible = 0
    for i,j in pairs:
        a,b = curves['NA'][i], curves['non-NA'][j]
        if any(m['clean_retention'] is None or m['clean_retention'] < .8 or m['emptied_relations'] for m in (a,b)):
            continue
        feasible += 1
        m = combine(a,b)
        key = (m['purity'], m['clean_retention'], -m['deletion_rate'], -GRID[i], -GRID[j])
        if best is None or key > best[0]:
            best = (key, {'NA': GRID[i], 'non-NA': GRID[j]}, m)
    if best is None:
        raise RuntimeError('No feasible configuration')
    return dict(config=best[1], fit_metrics=best[2], feasible_pairs=feasible, curves=curves)


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(variants=VARIANTS, grid=GRID, flank_tokens=3, min_keep=1,
                    objective='maximize overall retained purity; each of NA and non-NA clean retention >= .8; no observed relation emptied; tie: clean retention, fewer deletions, lower NA threshold, lower non-NA threshold',
                    folds='5; sha256(simple-ablation-v1: + normalized sentence text hash) modulo 5; all mentions of same text together; bags rebuilt separately inside each partition',
                    fixed='weights, biases, NA scaling, uncertainty/clutter/overlap, temperature, data and valid-span selection unchanged',
                    local_scope='only full-text keyword and boost matching become interval +/-3 whitespace tokens with target spans masked; between keywords unchanged; no distance-weight adjustment',
                    reference='old frozen global threshold .07 reported separately; fair baseline refitted under same CV and group constraints',
                    evaluation='OOF dev is exploratory because dev was previously inspected; test/test_noise are diagnostics after all dev fits frozen, not pristine holdouts',
                    go_no_go='Prefer a changed variant only if pooled OOF purity and non-NA noise recall both exceed the retuned baseline, with OOF clean retention >= .8 for each group and no relation emptied. Otherwise retain baseline. No test-based selection.',
                    environment=dict(python=sys.version,platform=platform.platform()),
                    source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'filter/model.py',ROOT/'filter/simple_variants.py',ROOT/'scripts/simple_ablation.py',ROOT/'data/arnor_benchmark_v2/arnor_dev.txt',ROOT/'data/arnor_benchmark_v2/arnor_test.txt',ROOT/'data/arnor_benchmark_v2/arnor_test_noise.txt']})
    write_json(out/'protocol.json', protocol)
    bags = load_dataset(ROOT/'data/arnor_benchmark_v2/arnor_dev.txt')[0]
    dev = {'global':score(bags), 'local':score(bags,LocalKeywordProvider())}
    assignments = {r['sentence_text_id']:fold_id(r['sentence_text_id']) for r in dev['global']}
    write_json(out/'sentence_folds.json', assignments)
    results = {}; frozen = {}; all_cv = {}
    for name,(scope,family) in VARIANTS.items():
        rows = dev[scope]; oof = []; folds = []
        for fold in range(5):
            fit_rows = [r for r in rows if assignments[r['sentence_text_id']]!=fold]
            val_rows = [r for r in rows if assignments[r['sentence_text_id']]==fold]
            assert {r['sentence_text_id'] for r in fit_rows}.isdisjoint(r['sentence_text_id'] for r in val_rows)
            fitted = fit(fit_rows,family)
            pred = [dict(r,fold=fold) for r in decisions(val_rows,fitted['config'])]
            oof.extend(pred);folds.append(dict(fold=fold,fit=fitted,validation=summarize(pred),fit_N=len(fit_rows),val_N=len(val_rows)))
            print(name, 'fold', fold, fitted['config'], flush=True)
        assert len(oof)==len(rows) and len({r['instance_id'] for r in oof})==len(rows)
        all_cv[name] = folds;results[name] = {'oof_dev':summarize(oof)}
        write_rows(out/f'oof_{name}.jsonl',oof)
        fitted = fit(rows,family); frozen[name] = dict(scope=scope, family=family, **fitted)
        results[name]['fit_dev'] = summarize(decisions(rows,fitted['config']))
        write_json(out/f'cv_{name}.json',folds)
    baseline = results['baseline']['oof_dev']
    eligible=[]
    for name in ['dual','local','combined']:
        m=results[name]['oof_dev']
        if (m['overall']['purity']>baseline['overall']['purity'] and m['non-NA']['recall']>baseline['non-NA']['recall'] and
            all(m[g]['clean_retention']>=.8 for g in ['NA','non-NA']) and not m['overall']['emptied_relations']):eligible.append(name)
    chosen=max(eligible,key=lambda n:results[n]['oof_dev']['overall']['purity']) if eligible else 'baseline'
    write_json(out/'frozen_configs.json',frozen)
    write_json(out/'dev_decision.json',dict(chosen=chosen,eligible=eligible,reason='Predeclared OOF gate; test metrics not yet computed'))
    # All dev selections have been persisted before reading test labels.
    for split in ['test','test_noise']:
        b=load_dataset(ROOT/f'data/arnor_benchmark_v2/arnor_{split}.txt')[0]
        scored={'global':score(b),'local':score(b,LocalKeywordProvider())}
        for name,(scope,_) in VARIANTS.items():
            pred=decisions(scored[scope],frozen[name]['config'])
            results[name][split]=summarize(pred);write_rows(out/f'{split}_{name}.jsonl',pred)
            # Remove dev text overlap BEFORE bag selection, not after decisions.
            subset=[r for r in scored[scope] if r['sentence_text_id'] not in assignments]
            results[name][split+'_no_dev_text']=summarize(decisions(subset,frozen[name]['config']))
        results.setdefault('old_frozen_reference',{})[split]=summarize(decisions(scored['global'],{'NA':.07,'non-NA':.07}))
    results['old_frozen_reference']['dev_full']=summarize(decisions(dev['global'],{'NA':.07,'non-NA':.07}))
    # Paired OOF switches explain changes without mining the diagnostic test set.
    base={r['instance_id']:r for r in map(json.loads,(out/'oof_baseline.jsonl').read_text().splitlines())}
    switches={}
    for name in ['dual','local','combined']:
        changed=[]
        for r in map(json.loads,(out/f'oof_{name}.jsonl').read_text().splitlines()):
            old=base[r['instance_id']]
            if old['keep']!=r['keep']:changed.append(dict(instance_id=r['instance_id'],relation=r['relation'],is_noise=r['is_noise'],baseline_keep=old['keep'],variant_keep=r['keep'],text=r['text'],h=r['h']['name'],t=r['t']['name'],baseline_raw=old['raw_score'],variant_raw=r['raw_score'],context=r['components'].get('lexical_context')))
        switches[name]=changed
    write_json(out/'oof_decision_changes.json',switches)
    write_json(out/'results.json',results)
    print('Done. OOF choice:',chosen,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'results/simple_ablation_20261014');a=p.parse_args();run(a.out)

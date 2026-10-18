"""Independently verify the final high-recall experiment and write its summary."""
import argparse,collections,hashlib,json
from pathlib import Path
from retest import ROOT
from logistic_filter import predict_noise
from simple_ablation import fold_id

def main(out):
 results=json.loads((out/'results.json').read_text());cfg=json.loads((out/'frozen_configs.json').read_text());checks={}
 for target,sets in results.items():
  outer=json.loads((out/f'outer_{target}.json').read_text())
  for split in ['oof','dev','test','test_noise']:
   for kind in ['logistic','raw']:
    rows=[json.loads(l) for l in (out/f'{split}_{kind}_{target}.jsonl').read_text().splitlines()]
    assert len(rows)==len({r['instance_id'] for r in rows});c=collections.Counter()
    for r in rows:
     if split=='oof':assert r['fold']==fold_id(r['sentence_text_id'])
     if kind=='logistic':
      model=outer[r['fold']]['model'] if split=='oof' else cfg[target]['logistic'];p=predict_noise([r],model)[0];assert abs(p-r['selection_score'])<1e-12;expected=p<r['threshold']
     else:expected=r['raw_score']>=r['threshold']
     assert r['keep']==expected
     if r['is_noise'] is not None:c[('FN' if r['keep'] else 'TP') if r['is_noise'] else ('TN' if r['keep'] else 'FP')]+=1
    key=f'{split}_{kind}';m=sets[key]['overall'];assert all(c[k]==m[k] for k in ['TP','FP','FN','TN']);checks[target+'_'+key]=dict(N=len(rows),kept=sum(r['keep'] for r in rows),counts=dict(c))
 # Verify historical inner-cache model/data dependencies, not only cache paths.
 previous=json.loads((ROOT/'results/logistic_20261014/protocol.json').read_text())
 for f,h in previous['sources'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
 for file in out.glob('*_tuning.json'):
  prefix=file.name.split('_target')[0]
  for candidate in json.loads(file.read_text()):
   cache=ROOT/f"results/logistic_20261014/{prefix}_C{candidate['C']}_inner_predictions.jsonl"
   assert hashlib.sha256(cache.read_bytes()).hexdigest()==candidate['cache_sha256']
 (out/'verification.json').write_text(json.dumps(checks,indent=2))
 pct=lambda x:'未定义' if x is None else f'{100*x:.2f}%'
 text='''# 最终调整：噪声召回优先

2026-10-14。目标从“保留≥95%的clean”改为“先达到高噪声召回，再尽量多留clean”。90%、95%、99%是展示取舍的三个召回目标，不是文献标准，也不是对新数据的保证。所有结果均保存，没有根据test挑选最好看的单点。原稿与此前实验保持不变。

每组NA/non-NA分别在拟合数据上达到目标召回，在满足目标的阈值中最大化clean保留率。取消每bag至少留一句、clean最低保留率和关系覆盖约束；报告删空关系。对照为raw规则阈值和L2逻辑回归，另有keep-all/drop-all。逻辑回归沿用九个规则特征及C={.01,.1,1,10}，复用此前已审计的内部折外概率（训练算法没有变化），仅按新目标重选C与阈值，再重拟合外层模型。

每个外层验证折完全不参与C和阈值选择。同句分组五折；全开发配置冻结后再评价旧test/test_noise。开发与测试数据都已被反复查看，因此结论属于探索性复核。模型不得声称取得了全新独立测试结果。

## 开发集折外：整体结果

|目标召回|方法|实际噪声召回|clean保留率|保留集纯度|保留/总数|
|---|---|---:|---:|---:|---|
'''
 for t,s in results.items():
  for kind in ['raw','logistic']:
   m=s['oof_'+kind]['overall'];c=checks[t+'_oof_'+kind];text+=f"|{t}%|{kind}|{pct(m['recall'])}|{pct(m['clean_retention'])}|{pct(m['purity'])}|{c['kept']}/{c['N']}|\n"
 for split in ['test','test_noise']:
  text+=f'\n## {split}：冻结后整体诊断\n\n|目标召回|方法|实际噪声召回|clean保留率|保留集纯度|保留/总数|\n|---|---|---:|---:|---:|---|\n'
  for t,s in results.items():
   for kind in ['raw','logistic']:
    m=s[split+'_'+kind]['overall'];c=checks[t+'_'+split+'_'+kind];text+=f"|{t}%|{kind}|{pct(m['recall'])}|{pct(m['clean_retention'])}|{pct(m['purity'])}|{c['kept']}/{c['N']}|\n"
 text+='\n## 非NA必须单列\n\n|集合|目标召回|逻辑回归实际召回|clean保留率|保留集纯度|TP/FP/FN/TN|\n|---|---|---:|---:|---:|---|\n'
 for split in ['oof','test','test_noise']:
  for t,s in results.items():
   m=s[split+'_logistic']['non-NA'];text+=f"|{split}|{t}%|{pct(m['recall'])}|{pct(m['clean_retention'])}|{pct(m['purity'])}|{m['TP']}/{m['FP']}/{m['FN']}/{m['TN']}|\n"
 text+='''
## 如何解释

- 90%目标配置在test上总体召回95.12%、纯度90.16%，留下488/4059条；其中440条clean、48条noise。原有效子集共有3076条clean，因此仅保留14.30%的clean，不能只写“去除95%噪声、纯度90%”而省略代价。
- 同一配置的非NA只留下63条（48clean、15noise），纯度76.19%；原非NA纯度75.74%。因此整体纯度收益不能泛化为强非NA关系理解。
- 95%目标配置的test整体纯度89.87%，反而略低于90%档。这说明删除更多噪声不保证剩余数据更纯，因为clean也在丢失。
- 99%档在test仅留1条clean，在test_noise没有保留已标注实例；“100%召回/纯度”在这种极小或空集上无实用优势。空集纯度明确为未定义。
- 原保守拒绝条件最多覆盖开发非NA噪声101/253≈39.92%，根本无法达到90%以上召回，因此没有把它伪装成可达到本轮目标的策略。它仍作为先前低误删工作点保留。
- 这轮可以支持“可解释特征＋可配置的删噪/数据量权衡”，不能支持“兼顾高召回与高数据保留”“超过神经基线”或“改善下游关系分类”。没有进行匹配的下游重训。

## 95%数字的区别

此前95%是我方实验选择的clean保留约束，不是原论文或行业标准；本轮95%是噪声召回目标。最终实际的score阈值另存在model_recall*.json中，三个概念不能混用。

## 复现和使用

依赖requirements-logistic-lock.txt；完整流程先生成此前嵌套实验的内部预测，再运行本轮。缓存复用前后通过来源及SHA-256核查，所有最终指标独立复算，模型概率重算与JSONL一致。32项测试通过。

    .venv-logreg/bin/python scripts/recall_frontier.py --out results/recall_new_run
    .venv-logreg/bin/python scripts/report_recall_frontier.py --out results/recall_new_run
    python3 filter/logistic_filter.py --model results/recall_frontier_20261014/model_recall90.json --input data/arnor_benchmark_v2/arnor_test.txt --output /private/tmp/recall90_decisions.jsonl

所有配置均允许删空bag；这三个预设不是自动采用的最佳模型。保留每个工作点，而不是把测试结果最漂亮的数字定成唯一成果。原始数据、旧论文与旧结果未覆盖。
'''
 (out/'噪声召回优先最终报告.md').write_text(text);print('All frontier metrics and cache provenance verified.')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'results/recall_frontier_20261014');a=p.parse_args();main(a.out)

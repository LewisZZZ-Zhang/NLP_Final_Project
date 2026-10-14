"""Audit nested split lineage, exported scores, and every final decision."""
import argparse,collections,hashlib,json,statistics,sys
from pathlib import Path
import numpy as np
from logistic_experiment import ROOT
from logistic_filter import FEATURES,feature_vector,predict_noise
from retest import score,load_dataset
from simple_ablation import fold_id


def read_rows(path):return [json.loads(l) for l in path.read_text().splitlines()]

def main(p):
 rows=score(load_dataset(ROOT/'data/arnor_benchmark_v2/arnor_dev.txt')[0]);by_id={r['instance_id']:r for r in rows}
 results=json.loads((p/'results.json').read_text());model=json.loads((p/'model.json').read_text());decision=json.loads((p/'dev_decision.json').read_text());outer=json.loads((p/'outer_summary.json').read_text());checks={};fit_count=0
 for path in p.glob('*_lineage.json'):
  lineage=json.loads(path.read_text());label=path.name.split('_C')[0]
  prohibited={r['instance_id'] for r in rows if label.startswith('outer') and fold_id(r['sentence_text_id'])==int(label[5:])}
  for fit in lineage:
   train_ids=set(fit['train_ids']);val_ids=set(fit['validation_ids']);assert not train_ids&val_ids;assert not (train_ids|val_ids)&prohibited
   train=[by_id[i] for i in fit['train_ids']];val=[by_id[i] for i in fit['validation_ids']]
   assert {r['sentence_text_id'] for r in train}.isdisjoint(r['sentence_text_id'] for r in val)
   X=np.array([feature_vector(r) for r in train]);assert np.allclose(X.mean(axis=0),fit['model']['mean'],atol=1e-12)
   expected=X.std(axis=0);expected[expected==0]=1;assert np.allclose(expected,fit['model']['scale'],atol=1e-12)
   fit_count+=1
 for part,key in [('oof','oof_dev'),('test','test'),('test_noise','test_noise')]:
  for variant in ['logistic','baseline','guarded']:
   rs=read_rows(p/f'{part}_{variant}.jsonl');assert len(rs)==len({r['instance_id'] for r in rs});c=collections.Counter()
   for r in rs:
    if variant=='logistic':
     assert r['keep']==(r['noise_score']<r['threshold'])
     if part=='oof':assert r['fold']==fold_id(r['sentence_text_id'])
    if r['is_noise'] is not None:c[('FN' if r['keep'] else 'TP') if r['is_noise'] else ('TN' if r['keep'] else 'FP')]+=1
   assert all(c[k]==results[variant][key]['overall'][k] for k in ['TP','FP','FN','TN'])
   checks[f'{part}_{variant}']=dict(N=len(rs),counts=dict(c),unique_ids=True)
  if part=='oof':
   for f in outer:
    rs=[r for r in read_rows(p/'oof_logistic.jsonl') if r['fold']==f['fold']]
    assert np.allclose(predict_noise(rs,f['model']),[r['noise_score'] for r in rs],atol=1e-12)
  else:
   rs=read_rows(p/f'{part}_logistic.jsonl');assert np.allclose(predict_noise(rs,model),[r['noise_score'] for r in rs],atol=1e-12)
  for current,old in [('baseline','baseline'),('guarded','guarded_rejection')]:
   prev=read_rows(ROOT/f'results/bag_rejection_20261014_run2/{part}_{old}.jsonl')
   assert {r['instance_id']:r['keep'] for r in prev}=={r['instance_id']:r['keep'] for r in read_rows(p/f'{part}_{current}.jsonl')}
 for file,digest in json.loads((p/'protocol.json').read_text())['sources'].items():assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==digest
 checks['lineage']=dict(inner_fits_verified=fit_count,train_validation_sentence_disjoint=True,outer_validation_excluded_from_tuning=True,scaler_means_scales_training_only=True,exported_predictions_reproduced=True)
 (p/'verification.json').write_text(json.dumps(checks,indent=2))
 pct=lambda x:'未定义' if x is None else f'{x*100:.2f}%'
 names={'baseline':'规则基线','guarded':'保守拒绝规则','logistic':'L2逻辑回归'}
 text='''# 基于规则特征的轻量监督过滤器

日期：2026-10-14。只训练小型逻辑回归，不使用 BERT，不改变现有规则特征提取。原始仓库、此前结果与默认模型保持不变。

## 结论

这版已训练完成，但未通过预设采用条件，不建议替换保守拒绝方案。开发外层折外非 NA 结果：逻辑回归 TP=22、FP=35、clean 保留率94.79%、噪声召回8.70%；保守规则 TP=43、FP=32、clean 保留率95.24%、噪声召回17.00%。逻辑回归没有满足95%目标，也没有超过规则的噪声识别率。

旧 test 上逻辑回归识别19条非 NA噪声、错删36条clean，clean保留率95.15%；比保守规则识别的13条更多，但误删也更多。test_noise识别60条非NA噪声、错删6条clean，但clean保留率93.94%，低于目标。因此不能用某个测试集更高的召回来推翻开发选择。

非NA外层ROC AUC为0.5353，说明在本协议下这组特征的线性组合区分能力较弱；它不证明这些特征对所有模型都无用。overlap在全部valid开发实例中恒为0；order几乎恒为1，boosts/uncertainty较稀疏。这些特征在当前数据中可提供的信息有限。学习权重并没有消除关系语义证据不足的问题。

本轮到此冻结，不继续根据测试结果加特征、扩大网格或降低clean保留目标。模型与可复现实验仍作为已完成的监督基线交付。

## 训练与验证协议

输入为 distance、between、keyword、order、boosts、uncertainty、clutter、overlap 和 NA 指示变量，共 9 维。人工 bias、最终 raw score、原文、实体名称、关系 one-hot、bag 大小均不输入。模型有独立学习的截距；这与沿用人工关系 bias 不同。

模型采用 StandardScaler + L2 LogisticRegression，lbfgs，C∈{.01,.1,1,10}，max_iter=2000、tol=1e-8、不做类别权重调整。标签 True=噪声。概率输出只是未经额外校准的模型估计，不能视为已验证的可信概率。

开发集 4241 条 valid、已标注实例用于训练；原 train 无人工噪声标签，不参与监督拟合。外层沿用同句文本哈希五折；每个外层训练部分再用剩余四个句子折作内部验证。所有标准化统计量和模型权重只由内部训练分区拟合。C 与 NA/non-NA 阈值均由合并的内部折外预测选择，随后在完整外层训练部分重拟合并评价外层验证折。

阈值分别在 NA、非 NA 中选取：从 0 到 1、步长 .01，并加入 >1 的全部保留选项；要求该组内部 clean 保留率≥95%，最大化噪声召回，并列优先纯度、clean 保留率、较高阈值。C 优先非 NA 噪声召回、非 NA 纯度，再比较总体内部 log loss，最后优先更强正则（更小 C）。删除规则是 noise_score≥threshold，不是旧 raw 的方向。允许删空 bag，没有关系覆盖硬约束，删空关系会完整报告。

规则基线和保守拒绝在同一外层验证实例上评价；保守规则阈值只用外层训练数据重新选择。它们的 NA 策略保持原样，逻辑回归则额外对 NA 也设置 95% 内部保留目标，因此应以分组指标解释差异，不能只看总体纯度。

完整开发集的最终模型通过五折内部预测选择参数，再在全部开发数据重拟合。模型、阈值、开发选择均先保存，之后才计算旧 test/test_noise 诊断。嵌套验证降低本轮调参泄漏，但不能消除多次查看开发数据对设计的影响；旧测试集也不重新成为独立留出集。

## 外层五折结果

|方案|非NA噪声召回|非NA clean保留率|非NA保留纯度|非NA TP/FP/FN/TN|总体纯度|
|---|---:|---:|---:|---|---:|
'''
 for n in names:
  m=results[n]['oof_dev'];g=m['non-NA'];text+=f"|{names[n]}|{pct(g['recall'])}|{pct(g['clean_retention'])}|{pct(g['purity'])}|{g['TP']}/{g['FP']}/{g['FN']}/{g['TN']}|{pct(m['overall']['purity'])}|\n"
 text+=f"\n预设候选条件是否通过：**{'通过' if decision['candidate_passed'] else '未通过'}**。要求合并外层非 NA clean 保留率≥95%，噪声召回高于保守规则，且非 NA 纯度不低于保守规则。本轮没有自动替换默认模型。\n"
 text+='\n### 每个外层折：不能只看均值\n\n|折|C|NA阈值|非NA阈值|非NA recall|非NA clean保留率|非NA纯度|\n|---|---:|---:|---:|---:|---:|---:|\n'
 for f in outer:
  m=f['model'];g=f['validation']['non-NA'];text+=f"|{f['fold']}|{m['C']}|{m['thresholds']['NA']}|{m['thresholds']['non-NA']}|{pct(g['recall'])}|{pct(g['clean_retention'])}|{pct(g['purity'])}|\n"
 text+='\n### NA 分组\n\n|方案|噪声召回|clean保留率|保留纯度|\n|---|---:|---:|---:|\n'
 for n in names:
  g=results[n]['oof_dev']['NA'];text+=f"|{names[n]}|{pct(g['recall'])}|{pct(g['clean_retention'])}|{pct(g['purity'])}|\n"
 for split in ['test','test_noise']:
  text+=f'\n## {split}：冻结后诊断\n\n|方案|非NA recall|非NA clean保留率|非NA纯度|非NA TP/FP/FN/TN|总体纯度|\n|---|---:|---:|---:|---|---:|\n'
  for n in names:
   m=results[n][split];g=m['non-NA'];text+=f"|{names[n]}|{pct(g['recall'])}|{pct(g['clean_retention'])}|{pct(g['purity'])}|{g['TP']}/{g['FP']}/{g['FN']}/{g['TN']}|{pct(m['overall']['purity'])}|\n"
 text+='''
测试质量指标只覆盖 valid span 子集；test_noise 的四条未知标签不进入质量分母。分关系、单实例/多实例、bag 删空情况以及去除开发相同文本后的诊断保存在 results.json。去重叠诊断中的 bag 大小沿用原输入批次；模型决策本身逐实例且不依赖 bag。

## 最终模型与解释

'''
 text+=f"最终 C={model['C']}；NA 删除阈值={model['thresholds']['NA']}，非 NA 删除阈值={model['thresholds']['non-NA']}。最终模型迭代次数={model['iterations']}。\n\n"
 text+='正系数表示该标准化特征增大时，模型更倾向预测噪声；负系数相反。系数是条件关联，不是因果作用。距离与 between 等特征可能相关，不能仅按绝对系数比较原始特征重要性。\n\n|特征|标准化系数|训练均值|标准化尺度|\n|---|---:|---:|---:|\n'
 for name,w,mean,scale in zip(FEATURES,model['coefficients'],model['mean'],model['scale']):text+=f'|{name}|{w:.6f}|{mean:.6f}|{scale:.6f}|\n'
 text+=f"\n学习的截距为 {model['intercept']:.6f}。\n"
 text+='\n### 评分区分能力（外层折外预测）\n\n|分组|ROC AUC|Average precision|Brier score|\n|---|---:|---:|---:|\n'
 for g,v in results['logistic']['oof_dev']['score_diagnostics'].items():text+=f"|{g}|{v['roc_auc']:.4f}|{v['average_precision']:.4f}|{v['brier']:.4f}|\n"
 text+='\n### 关系覆盖\n\n'
 for split in ['oof_dev','test','test_noise']:
  m=results['logistic'][split];text+=f"- {split}：全部 bag {m['bag_coverage']['total']}，删空 {m['bag_coverage']['emptied']}；完全删空的关系：{m['overall']['emptied_relations'] or '无'}。\n"
 text+='外层个别折的关系删空情况在 outer_summary.json 中保留；合并后没有删空某关系不代表每折都覆盖该关系。\n'
 timing=json.loads((p/'timing.json').read_text());text+=f"\n## CPU 成本\n\n全流程（含规则对照重选阈值、输出）{timing['full_experiment_seconds']:.2f}s；最终一次逻辑回归拟合 {timing['final_fit_seconds']:.4f}s；已提取的4241条特征用纯Python导出模型推理中位数 {statistics.median(timing['inference_4241_feature_rows_seconds']):.4f}s。推理数值不含特征提取/读写，不能拿它与旧端到端过滤时间比较。线程环境变量设为1，无神经训练或GPU。\n"
 text+=f"\n## 验证与复现\n\n独立核对了 {fit_count} 次内部拟合的训练/验证句子隔离、外层验证排除及标准化均值/尺度；所有最终混淆矩阵独立复算；JSON模型与scikit-learn预测一致。规则对照与上一轮逐实例决定相同。完整测试结果见 tests.log。\n"
 text+='''
训练依赖锁定在 requirements-logistic-lock.txt；独立环境 .venv-logreg，不修改系统环境。复现实验需新目录：

    .venv-logreg/bin/python -m unittest discover -s tests -v
    .venv-logreg/bin/python scripts/logistic_experiment.py --out results/logistic_new_run
    .venv-logreg/bin/python scripts/report_logistic.py --out results/logistic_new_run

导出的 model.json 只含均值、尺度、九个系数、截距、阈值与元数据。无需 scikit-learn 即可推理：

    python3 filter/logistic_filter.py --model results/logistic_20261014/model.json --input data/arnor_benchmark_v2/arnor_test.txt --output /private/tmp/logistic_predictions.jsonl

输出保留所有已评分实例及决策，不会静默丢掉被删除的记录；输出文件存在时拒绝覆盖。原始 invalid/ambiguous 输入仍需单独核查，不会被当作模型噪声判定。

此模型应称“基于规则特征的轻量监督过滤器”，不能再称完全无需训练。正式更新应用默认策略或申请材料前，应考虑重复探索和缺少新独立标签的限制。

方法参考：[scikit-learn Pipeline防止预处理泄漏](https://scikit-learn.org/stable/modules/compose.html)、[LogisticRegression官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)。
'''
 (p/'轻量逻辑回归实验报告.md').write_text(text);print('Independent nested-CV verification passed;',fit_count,'inner fits audited.')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'results/logistic_20261014');a=p.parse_args();main(a.out)

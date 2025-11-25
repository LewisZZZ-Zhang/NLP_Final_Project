/tmp，/out 两个文件夹是用来存临时文件的，里面的文件不会同步到git

数据集：
https://www.kaggle.com/datasets/daishinkan002/new-york-times-relation-extraction-dataset?resource=download&utm_source=chatgpt.com
放在\data\NYTRED里面
├── data
│   └── NYTRED
│       ├── test.json
│       ├── train.json
│       └── valid.json


数据集2 arnor_dataset-2.0.0 
https://github.com/PaddlePaddle/Research/blob/master/NLP/ACL2019-ARNOR/README.md
该数据集为论文模型ARNOR所使用的训练集的优化版本， 特点是在test数据集中直接存在手动标记的is_nosie标签
可以用于直接测试attention block过滤噪声的能力。
需要自行下载并放在\data\arnor_dataset-2.0.0里面
下载完成后还需先运行scripts/convert_arnor_to_opennre.py 随后代码就能正常运行
以下是文件夹结构
├── data
│   └── arnor——dataset-2.0.0
│       ├── dev.json          用于参数调试
│       ├── test_nosie.json   用于filter表现评估
│       ├── test.json         用于RC模型表现评估
│       └── train.json        用于RC/filter模型训练


以下附有原json文件的数据结构：    
{
    "sentText": "The source sentence text",
    "relationMentions": [
                            {
                                "em1Text": "The first entity in relation",
                                "em2Text": "The second entity in relation",
                                "label": "Relation label",
                                "is_noise": false # only occur in test set
                            },
                            ...
                        ],
    "entityMentions":   [
                            {
                                "text": "Entity words",
                                "label": "Entity type",
                                ...
                            },
                            ...
                        ]
    ...
}


How to run OpenNRE:
--Environment Preparation
pip install -r requirements.txt
python setup.py install

--Downloading the dataset
cd ./benchmark
bash download_nyt10m.sh

Training:
python3 example/train_bag_cnn.py \
    --metric auc \
    --dataset nyt10m \
    --batch_size 160 \
    --lr 0.1 \
    --weight_decay 1e-5 \
    --max_epoch 100 \
    --max_length 128 \
    --seed 42 \
    --encoder pcnn \
    --aggr att


**ARNOR → OpenNRE conversion**
- **Script:** `OpenNRE/scripts/convert_arnor_to_opennre.py`
- **What it does:** converts Arnor JSON (one JSON object per line) into OpenNRE line-by-line format by emitting one OpenNRE-style record per `relationMention`. The script preserves the original `is_noise` flag by adding `"is_noise": true|false` to each output record.
- **Default input:** `data/arnor_dataset-2.0.0/test.json`
- **Default outputs:**
    - `OpenNRE/benchmark/arnor_test.txt`  (one JSON dict per line; fields: `text`, `h`/`t` with `pos` char spans, `relation`, `is_noise`)
    - `OpenNRE/benchmark/arnor_rel2id.json` (automatic rel2id; `NA` -> 0)
- **How entity spans are generated:** the script tries several heuristics to locate entity surface text in the sentence (direct find, case-insensitive, whitespace-normalized, token-sequence match). If a span cannot be found a fallback `(0,0)` span is written and a warning is printed during conversion; you should inspect such warnings and adjust if precise spans are required.

Run the converter (from project root):
```bash
python3 OpenNRE/scripts/convert_arnor_to_opennre.py
```

Inspect the first few converted examples:
```bash
head -n 5 OpenNRE/benchmark/arnor_test.txt
cat OpenNRE/benchmark/arnor_rel2id.json | jq .
```

Mapping to OpenNRE benchmark `rel2id`:
- A mapping file was created to compare Arnor relations to the `nyt10m` relation vocabulary:
    - `OpenNRE/benchmark/arnor_to_nyt10m_mapping.json`
- The mapping uses exact-match first, then simple heuristics (small underscore/word differences). Review the mapping file and adjust if you prefer alternative mappings.

Quick sanity note:
- The OpenNRE encoders expect `h.pos` and `t.pos` to be character spans inside `text`. If you plan to train with BERT encoders, verify a small subset loads correctly with your chosen encoder/tokenizer.

Contact / Next steps:
- If you want, I can (a) run a short data-loader sanity check that imports the file into OpenNRE's `SentenceREDataset`, or (b) produce a CSV-style mapping with confidence scores for manual review. Tell me which to do next.


Citation
@inproceedings{han-etal-2019-opennre,
    title = "{O}pen{NRE}: An Open and Extensible Toolkit for Neural Relation Extraction",
    author = "Han, Xu and Gao, Tianyu and Yao, Yuan and Ye, Deming and Liu, Zhiyuan and Sun, Maosong",
    booktitle = "Proceedings of EMNLP-IJCNLP: System Demonstrations",
    year = "2019",
    url = "https://www.aclweb.org/anthology/D19-3029",
    doi = "10.18653/v1/D19-3029",
    pages = "169--174"
}



================================
数据准备：
cd data
bash download_nyt10.sh
bash download_nyt10m.sh

数据集说明：
（https://github.com/thunlp/OpenNRE.git）
（https://opennre-docs.readthedocs.io/en/latest/get_started/benchmark.html#bag-level-relation-extraction）
| 项目                        | NYT10                                                                      | NYT10m                                                                                |
| -------------------------- | ---------------------------------------------------------------------------| ------------------------------------------------------------------------------------- |
| 关系类别数 (#class)          | 53 个关系类别。([opennre-docs.readthedocs.io][1])                            | 25 个关系类别。([akbc.ws][2])                                                           |
| 实例（bags 或句子／实体对）规模 | 694,491 个实例。([opennre-docs.readthedocs.io][1])                          | 474,059 个训练实例（或总实例）约数。([cse.iitd.ac.in][3])                                  |
| 测试集标注质量                | 多为远监督自动标注（noisy labels）——原始 NYT10 测试集没有人工清洗。([akbc.ws][2]) | 测试集为 **人工标注**（manually annotated test set），目的是提升评测可信度。([akbc.ws][2])     |
| 训练／验证集设置              | 原始训练集为远监督构造；没有专门人工清洗验证集。([cse.iitd.ac.in][3])              | 鼓励有验证集，并且关系类别在训练与测试之间保持一致性。([akbc.ws][2])                           |
| 任务目标（适用性）            | 更“经典”、大规模，但噪声更多，测试集可能更偏向 “held-out” 远监督情境。              | 更严格用于评测：测试集人工标注，更适合作为“高质量”benchmark。                                  |
| 推荐使用场景                 | 若你想使用大规模数据训练，探索模型在大量远监督实例上的表现。                         | 若你想做模型评估、benchmark 比较、追求更可靠的评价结果。                                     |


python3 filter/main.py --split train --limit-bags 5 --policy topk --topk 1 --output tmp/nyt10m_train_filtered_sample.txt

in main:
Implemented a standalone NYT10m filter CLI (filter/main.py) that mirrors the proposal’s attention-filter stage. It reads a split (nyt10m_train/val/test.txt), groups sentences into bags by (relation, head_id, tail_id), and assigns each sentence a heuristic attention score using relation-specific lexical cues, entity distance, and context-based features to mimic attention weights.
Added three filtering policies covering the proposal’s strategies—absolute threshold, top‑k, and relative margin—with a guaranteed minimum number of kept sentences per bag plus hooks to ingest external attention files. The tool writes a new JSONL split (optionally with attention scores embedded) and reports per-relation statistics so you can compare pre/post bag sizes and removal rates.
CLI options expose dataset paths, policy parameters, temperature, bag limits for debugging, and JSON summary output, making it easy to iterate on filter settings before retraining OpenNRE models.
main的更多使用例在main.py的最底下



===================

下面是对 main.py 的整理与概述（中文）：

**概要**
- 目的：对 OpenNRE / NYT10m 风格的 JSONL 数据集做“轻量注意力式”过滤——把句子聚成袋（bag），为每句计算注意力分数，再按策略筛选并输出过滤后的 JSONL。
- 输入/输出：读取每行一个 JSON 的记录（OpenNRE 格式），输出为 JSONL（可选择在保留记录上附加 attention_score 字段）。

**主要功能点**
- **分袋规则**: 使用关系 + head id + tail id 组成袋 id（函数 `make_bag_id`）。
- **句子表示**: `SentenceInstance` 保存 relation、原文、lower 文本、实体名、实体 span、原始数据等，并提供 `between_text()`（实体间文本）缓存。
- **实体 span 提取**: `safe_extract_span` 做“稳健”提取：优先用记录里的 `pos`，否则在正文中搜索实体表面词。
- **启发式注意力**: `HeuristicAttentionProvider`
  - 使用关系特定的词表和规则（`RELATION_HINTS`）作为轻量线索。
  - 分数组成（在 `_score_sentence` 中加权组合）：
    - 距离得分（实体中心字符距离越近越好，权重 0.35）
    - between（实体间文本长度，鼓励简短上下文，权重 0.25）
    - 关键词匹配（relation-specific keywords / between-keywords，权重 0.3）
    - 实体顺序偏好（preferred_order，权重 0.1）
    - relation bias / boosts（关系偏置与额外小幅提升）
  - 对 NA 关系的分数会乘以 0.4（降低其权重）。
  - 将 raw 分数通过温度 softmax（`temperature`）转成归一化 attention 权重。
- **外部注意力导入**: `ExternalAttentionProvider` 支持加载外部 JSONL（每行带 `bag_id` + `scores`/`attentions`），若长度不匹配或无条目则回退到启发式提供者。
- **过滤策略（Policy）**:
  - `AbsoluteThresholdPolicy`：按阈值保留（不足 min_keep 则保留 top min_keep）。
  - `TopKPolicy`：每袋保留 top-K。
  - `RelativeMarginPolicy`：保留分数在 top_score - margin 之上的句子（并保证 min_keep）。
  - `DropBottomPolicy`：丢弃最差的 K 条（保留至少 min_keep）。
  - `DropPercentPolicy`：丢弃最低 X%（保证 min_keep）。
- **统计与噪声评估**: `StatsTracker`
  - 跟踪处理的袋数、句子数、按关系的保留/删除统计。
  - 可选的噪声评估（`--evaluate-noise`）：如果数据含 `is_noise` 字段，会把“被丢弃=预测为噪声”与真实标签比对，统计 TP/FP/FN/TN 并计算 precision/recall/f1/accuracy。
- **CLI/运行参数**:
  - 数据、分割与文件路径：`--dataset-root`, `--split`, `--input`, `--output`
  - 策略与超参：`--policy`（`absolute/topk/relative/drop/drop_pct`）、`--threshold`、`--topk`、`--margin`、`--drop-bottom`、`--drop-percent`、`--min-keep`
  - 注意力控制：`--temperature`, `--attention-file`, `--annotate-attention`
  - 评估/报告：`--evaluate-noise`, `--report`
  - 其他：`--limit-bags`（调试用只读前 N 个袋）、`--log-level`
- **IO 与行为**:
  - 按袋遍历，获取 attention scores，策略 `select` 出保留掩码，写入输出文件（每条 JSONL 一行）。
  - 若 `--annotate-attention`，在保留的 record 中加入 `attention_score` 字段（4-6 小数位）。
  - 最后打印并（可选）保存统计摘要 JSON（`--report`）。

**实现细节/注重点**
- `RELATION_HINTS` 列举了很多常见关系的触发关键词、实体间关键词、偏好顺序与 bias/boost 值，便于可解释的启发式打分。
- 得分计算使用了手工设定的权重（0.35/0.25/0.3/0.1），并用 clamp 限制到合适范围。
- Softmax 温度可调，避免单条句子独占权重或过于平滑。
- 对 malformed JSON 行有容错（跳过并记录 skipped）。
- 对外部 attention 文件做长度匹配与归一化检查，不匹配时回退到启发式提供者。

**使用示例（文件底部已有）**
- 仅对前 5 个袋做 topk=1 筛选并输出样本：
  - python3 main.py --split train --limit-bags 5 --policy topk --topk 1 --output nyt10m_train_filtered_sample.txt
- 带噪声评估：
  - python3 main.py --input arnor_test.txt --policy drop_pct --drop-percent 0.5 --limit-bags 10 --evaluate-noise --output arnor_filtered.txt

**总结性评价**
- 这是一个面向 distant supervision 数据集的、以可解释启发式信号为核心的轻量过滤工具：
  - 优点：不依赖深度模型、易理解、可用外部 attention 接口融合其他方法、支持多种过滤策略与噪声评估。
  - 局限：启发式规则可能对某些关系或语域泛化不足；权重与关键词需人工调整或用数据驱动优化。



11/23 update
文件结构更新后 需先运行scripts/convert_arnor_to_nyt10m.py 随后代码就能正常运行
现在arnor dataset所有非原生文件均放在data/arnor_benchmark文件夹内 
并且转化后的文件有2个版本:
arnor_dev/train/test.txt: 仅包含arnor与nyt10m所共有的17中relation
arnor_dev/train/test_full.txt: 包含arnor数据集中的全部26中relation
具体relation种类请参考data/arnor_benchmark/common_relations.json


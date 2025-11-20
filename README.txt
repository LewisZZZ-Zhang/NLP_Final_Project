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
以下附有数据结构：
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

数据集说明：（https://opennre-docs.readthedocs.io/en/latest/get_started/benchmark.html#bag-level-relation-extraction）
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
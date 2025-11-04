数据集：
https://www.kaggle.com/datasets/daishinkan002/new-york-times-relation-extraction-dataset?resource=download&utm_source=chatgpt.com
放在\data\NYTRED里面
├── data
│   └── NYTRED
│       ├── test.json
│       ├── train.json
│       └── valid.json


关于数据集 可以参考 https://github.com/iesl/state-change-re 
这里面都是已经人工标注好的关系实体 大致结构为{doc_id:, sentence_id: entity:},
对应的文件需自行下载。
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
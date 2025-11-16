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
cd benchmark
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
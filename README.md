# **judicial_competition**
------------------------------

## **语言**
-----------
Python 3.5<br>


## **依赖库**
----------
numpy==1.14.2<br>
jieba==0.39<br>
pandas==0.22.0<br>
tensorflow==1.8.0<br>
Keras==2.1.6<br>
scikit-learn==0.19.1<br>

## **比赛详情**
![比赛介绍](/pics/competition.png)<br>
为了促进法律智能相关技术的发展，在最高人民法院信息中心、共青团中央青年发展部的指导下，中国司法大数据研究院、中国中文信息学会、中电科系统团委联合清华大学、北京大学、中国科学院软件研究所共同举办“2018中国‘法研杯’法律智能挑战赛（CAIL2018）”。<br>
官方网站：[http://cail.cipsc.org.cn/](http://cail.cipsc.org.cn/ "悬停显示")<br>
域名：[http://180.76.238.177/](http://180.76.238.177/ "悬停显示")<br>
GitHub：[https://github.com/thunlp/CAIL/"悬停显示"]

## **任务介绍**
--------------------
* __任务一 `罪名预测`：根据刑事法律文书中的案情描述和事实部分，预测被告人被判的罪名__<br>
* __任务二 `法条推荐`：根据刑事法律文书中的案情描述和事实部分，预测本案涉及的相关法条__<br>
* __任务三 `刑期预测`：根据刑事法律文书中的案情描述和事实部分，预测被告人的刑期长短__<br>

## **数据**
--------------------
这次采用的是第一阶段`CAIL2018-Small`数据，包括19.6万份文书样例，包括15万训练集，1.6万验证集和3万测试集。<br>
```
{'fact': '昌宁县人民检察院指控，2014年4月19日下午16时许，被告人段某驾拖车经过鸡飞乡澡塘街子，......',
 'meta': {'accusation': ['故意伤害'],
  'criminals': ['段某'],
  'punish_of_money': 0,
  'relevant_articles': [234],
  'term_of_imprisonment': {'death_penalty': False,
   'imprisonment': 12,
   'life_imprisonment': False}}}
```
`fact`:


共涉及202条罪名[罪名](/good/accu.txt)，183条[法条](/good/law.txt)，刑期长短包括0-25年、无期、死刑。<br>

## **模块简介**
这个模块主要包含三大块：
1. 数据预处理 data preprocessing
2. 模型 model
3. 预测 precditor

由于时间和任务的关系，这一次我只利用了第一阶段的数据实现了`罪名预测`和`刑期预测`两个任务，两个任务使用相同的模型结构。

## **数据预处理**
--------------------
数据预处理的功能主要包含在`data_utils`中，包括数据预处理各类功能函数集合[data_processing.py](/data_utils/data_processing.py)和对数据进行实际预处理的数据准备模块[data_preparation.py](/data_utils/data_preparation.py)。<br>
数据预处理要点：
1. 对文本进行分词进行分词，只保留长度大于1的词语（即去除单个字的）；
2.


## **模型**
--------------------
  在这个项目中我主要使用了CNN和TextCNN去做，对比起RNN，CNN在时间成本上更有优势





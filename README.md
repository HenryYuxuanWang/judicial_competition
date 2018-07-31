# **judicial_competition**
------------------------------

## **语言**
-----------
Python 3.5<br>
[![](https://img.shields.io/badge/Python-3.5-blue.svg)](https://www.python.org/)<br>


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
官方网站:  [http://cail.cipsc.org.cn](http://cail.cipsc.org.cn "悬停显示")<br>
域名:     [http://180.76.238.177](http://180.76.238.177 "悬停显示")<br>
GitHub:  [https://github.com/thunlp/CAIL ](https://github.com/thunlp/CAIL "悬停显示")

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
__*字段及意义:*__
* **fact**: 事实描述
* **meta**: 标注信息，标注信息中包括:
	* **criminals**: 被告(数据中均只含一个被告)
	* **punish\_of\_money**: 罚款(单位：元)
	* **accusation**: 罪名
	* **relevant\_articles**: 相关法条
	* **term\_of\_imprisonment**: 刑期
		刑期格式(单位：月)
		* **death\_penalty**: 是否死刑
		* **life\_imprisonment**: 是否无期
		* **imprisonment**: 有期徒刑刑期


共涉及202条[罪名](/good/accu.txt)，183条[法条](/good/law.txt)，刑期长短包括0-25年、无期、死刑。<br>

## **模块简介**
这个模块主要包含三大块：
1. 数据预处理 [data_utils](/data_utils)
2. 模型 [model](/model)
3. 预测 [precditor](/python_sample/predictor)

由于时间和任务的关系，这一次我只利用了第一阶段的数据实现了`罪名预测`和`法条推荐`两个任务，两个任务使用相同的模型结构。

## **数据预处理**
--------------------
数据预处理的功能主要包含在`data_utils`中，包括数据预处理各类功能函数集合[data_processing.py](/data_utils/data_processing.py)和对数据进行实际预处理的数据准备模块[data_preparation.py](/data_utils/data_preparation.py)。<br>

* __*基本流程*__：
1. 分别对train、test和valid数据进行分词并清洗；
2. 输入分好词的结果，使用keras的数据预处理工具把词语列表转化为词典，取频率最高的前40000个词语（数值可自定义）。事实证明词典大小对结果影响也是相当大的。
3. 根据词典，利用`texts_to_sequences`功能把词语列表转为序列（数字）列表，不在词典中的词语去掉；
4. 序列的长度固定为400（也可自定义，对后续的结果也是有一定的影响），利用`pad_sequences`对序列进行截断（长度大于400）或补全（长度少于400的补0）。

* __*数据预处理要点*__：
1. 对文本进行分词进行分词，只保留长度大于1的词语（即去除单个字的）；
2. 部分案情陈述中都有的涉案金额，但金额数量比较零散，不同意，容易导致在分词后建立词典时被筛掉，所以需要对涉案金额进行化整处理；
即预先订好固定的金额区间，如“50, 100, 200, 500, 800, 1000, 2000, 5000”，然后把处于对应区间的金额转化为固定的金额数值；
3. 去掉部分停用词，查阅了部分案情陈述后发现，大部分的案情陈述都涉及相同或类似的词语，如“某某，某某乡，某某县，被告人，上午，下午”等。
这类词语词频相当高，需要把他们去掉，以免影响对数据进行干扰。


## **模型**
--------------------
在这个项目的时候，对RNN和LSTM还不是很了解，所以主要使用了CNN和TextCNN去做，而且对比起RNN，CNN在时间成本上更有优势。<br>

__*CNN模型结构*：__<br>
![](/pics/cnn_filter3.png)

__*TextCNN模型结构*：__<br>
![](/pics/textcnn_filter345.png)





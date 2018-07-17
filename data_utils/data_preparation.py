from data_utils.data_processing import data_process
from keras.preprocessing.sequence import pad_sequences
import json
import pickle
import numpy as np

##################################################
# Configuration
num_words = 40000  # 词库的大小
max_document_length = 400  # 每一条犯罪事实最大的长度
pad_style = 'post'  # padding的方式，post为后，pre为前

##################################################
# 处理training data
# data_train
train_data_process = data_process()

train_data_process.get_data('./good/data_train.json')  # 获取训练数据

# 把train data中的fact取出
train_data_process.decompose_data('fact')
data_train_text = train_data_process.decomposition['fact']

# 分词
data_train_text_seg = train_data_process.segmentation(data_train_text, cut=True, word_len=1,
                                                      path='./seg/data_train_seg.json',
                                                      replace_money_value=True,
                                                      stopword=True)

# 导入分词文本
with open('./seg/data_train_seg.json', 'r') as f:
    data_train_seg = json.load(f)

# 保存长度为2以上的
data_train_seg_new = train_data_process.segmentation(data_train_text_seg, cut=False, word_len=2,
                                                     path='./seg/data_train_seg_new.json')

with open('./seg/data_train_seg_new.json', 'r') as f:
    data_train_seg_new = json.load(f)

train_data_process.text2num(text_list=data_train_seg_new, num_words=num_words)
tokenizer = train_data_process.tokenizer
with open('./variables/tokenizer_%d.pkl' % (num_words), mode='wb') as f:
    pickle.dump(tokenizer, f)

train_num_sequence = train_data_process.num_sequence  # 把原文本转成数字序列
train_pad_seg = pad_sequences(train_num_sequence, maxlen=max_document_length, padding=pad_style)

# 储存训练集
np.save('./variables/pad_sequences/train_pad_%s_%s.npy' % (max_document_length, num_words), train_pad_seg)
# 导入训练集
train_pad_seg = np.load('./variables/pad_sequences/train_pad_%s_%s.npy' % (max_document_length, num_words))

# 标签处理
# 取出law和accu数据
# relevant articles
label_type = 'relevant_articles'  # 处理的标签
train_data_process.decompose_data('law')  # decompose data - 取出原始数据中的每一部分数据, 主要为law/fact/accusation/
train_law_label = train_data_process.decomposition['law']
train_data_process.get_label_collection(label_type='law')
train_law = train_data_process.transform_label(labels=train_law_label, label_type='law')
np.save('./variables/labels/train_one_hot_%s.npy' % (label_type), train_law)
train_law = np.load('./variables/labels/train_one_hot_%s.npy' % (label_type))

# accusation
label_type = 'accusation'  # 处理的标签
train_data_process.decompose_data('accu')  # decompose data - 取出原始数据中的每一部分数据, 主要为law/fact/accusation/
train_accu_label = train_data_process.decomposition['accu']
train_data_process.get_label_collection(label_type='accu')
train_accu = train_data_process.transform_label(labels=train_accu_label, label_type='accu')
np.save('./variables/labels/train_one_hot_%s.npy' % (label_type), train_accu)
train_accu = np.load('./variables/labels/train_one_hot_%s.npy' % (label_type))

##################################################
# 处理test data
# data_test
test_data_process = data_process()

test_data_process.get_data('./good/data_test.json')

# 把train data中的text取出
test_data_process.decompose_data('fact')
data_test_text = test_data_process.decomposition['fact']

# 分词
data_test_seg = test_data_process.segmentation(data_test_text, word_len=1,
                                               path='./seg/data_test_seg.json',
                                               replace_money_value=True,
                                               stopword=True)

with open('./seg/data_test_seg.json', 'r') as f:
    data_test_seg = json.load(f)

# 取出长度大于2的
data_test_seg_new = test_data_process.segmentation(data_test_seg, cut=False, word_len=2,
                                                   path='./seg/data_test_seg_new.json')

with open('./seg/data_test_seg_new.json', 'r') as f:
    data_test_seg_new = json.load(f)

# 导入tokenizer
with open('./variables/tokenizer_%d.pkl'%(num_words), 'rb') as f:
    tokenizer = pickle.load(f,encoding='utf-8')


test_data_process.text2num(text_list=data_test_seg_new, tokenizer=tokenizer)
test_num_sequence = test_data_process.num_sequence
test_pad_seg = pad_sequences(test_num_sequence, maxlen=max_document_length, padding=pad_style)

# 储存测试集
np.save('./variables/pad_sequences/test_pad_%s_%s.npy' % (max_document_length, num_words), test_pad_seg)
# 导入测试集
test_pad_seg = np.load('./variables/pad_sequences/test_pad_%s_%s.npy' % (max_document_length, num_words))

# 标签处理
# relevant_articles
label_type = 'relevant_articles'  # 处理的标签
test_data_process.decompose_data('law')
test_law_label = test_data_process.decomposition['law']
test_data_process.get_label_collection(label_type='law')
test_law = test_data_process.transform_label(labels=test_law_label, label_type='law')
np.save('./variables/labels/test_one_hot_%s.npy' % (label_type), test_law)
test_law = np.load('./variables/labels/test_one_hot_%s.npy' % (label_type))

# accusation
label_type = 'accusation'  # 处理的标签
test_data_process.decompose_data('accu')  # decompose data - 取出原始数据中的每一部分数据, 主要为law/fact/accusation/
test_accu_label = test_data_process.decomposition['accu']
test_data_process.get_label_collection(label_type='accu')
test_accu = test_data_process.transform_label(labels=test_accu_label, label_type='accu')
np.save('./variables/labels/test_one_hot_%s.npy' % (label_type), test_accu)
test_accu = np.load('./variables/labels/test_one_hot_%s.npy' % (label_type))

###########################################################################################
# 验证集validation_set
# data_valid
valid_data_process = data_process()

valid_data_process.get_data('./good/data_valid.json')

# 把valid data中的fact取出
valid_data_process.decompose_data('fact')
data_valid_text = valid_data_process.decomposition['fact']

# 分词
data_valid_seg = valid_data_process.segmentation(data_valid_text, word_len=1,
                                                 path='./seg/data_valid_seg.json',
                                                 replace_money_value=True,
                                                 stopword=True)

with open('./seg/data_valid_seg.json', 'r') as f:
    data_valid_seg = json.load(f)

# 取出长度大于2的词语
data_valid_seg_new = valid_data_process.segmentation(data_valid_seg, cut=False, word_len=2,
                                                     path='./seg/data_valid_seg_new.json')

with open('./seg/data_valid_seg_new.json', 'r') as f:
    data_valid_seg_new = json.load(f)

# 导入tokenizer
with open('./variables/tokenizer_%d.pkl' % (num_words), 'rb') as f:
    tokenizer = pickle.load(f)

valid_data_process.text2num(text_list=data_valid_seg_new, tokenizer=tokenizer)
valid_num_sequence = valid_data_process.num_sequence
valid_pad_seg = pad_sequences(valid_num_sequence, maxlen=max_document_length, padding=pad_style)

# 储存测试集
np.save('./variables/pad_sequences/valid_pad_%s_%s.npy' % (max_document_length, num_words), valid_pad_seg)
# 导入测试集
valid_pad_seg = np.load('./variables/pad_sequences/valid_pad_%s_%s.npy' % (max_document_length, num_words))

# 标签处理
# relevant_articles
label_type = 'relevant_articles'  # 处理的标签
valid_data_process.decompose_data('law')
valid_law_label = valid_data_process.decomposition['law']
valid_data_process.get_label_collection(label_type='law')
valid_one_hot = valid_data_process.transform_label(labels=valid_law_label, label_type='law')
np.save('./variables/labels/valid_one_hot_%s.npy' % (label_type), valid_one_hot)
valid_law = np.load('./variables/labels/valid_one_hot_%s.npy' % (label_type))

# accusation
label_type = 'accusation'  # 处理的标签
valid_data_process.decompose_data('accu')  # decompose data - 取出原始数据中的每一部分数据, 主要为law/fact/accusation/
valid_accu_label = valid_data_process.decomposition['accu']
valid_data_process.get_label_collection(label_type='accu')
valid_accu = valid_data_process.transform_label(labels=valid_accu_label, label_type='accu')
np.save('./variables/labels/valid_one_hot_%s.npy' % (label_type), valid_accu)
valid_accu = np.load('./variables/labels/valid_one_hot_%s.npy' % (label_type))

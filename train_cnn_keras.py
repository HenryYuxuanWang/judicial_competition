from keras.models import Model
from keras.layers import Dense, Embedding, Input, Dropout, Activation
from keras.layers import Conv1D, BatchNormalization, GlobalMaxPooling1D
import pandas as pd
import numpy as np
from data_utils.evaluation import *
from sklearn.metrics import f1_score
from keras.utils.vis_utils import plot_model

embedding_dim = 256  # embedding layer dimension is fixed to 256
num_words = 40000
maxlen = 400
label_type = 'accusation'

##################################################
# data and label
train_fact_pad_seq = np.load('./variables/pad_sequences/train_pad_%d_%d.npy' % (maxlen, num_words))
valid_fact_pad_seq = np.load('./variables/pad_sequences/valid_pad_%d_%d.npy' % (maxlen, num_words))
test_fact_pad_seq = np.load('./variables/pad_sequences/test_pad_%d_%d.npy' % (maxlen, num_words))

train_labels = np.load('./variables/labels/train_one_hot_%s.npy' % (label_type))
valid_labels = np.load('./variables/labels/valid_one_hot_%s.npy' % (label_type))
test_labels = np.load('./variables/labels/test_one_hot_%s.npy' % (label_type))

# label list 标签的类别以及一共有多少类
set_labels = np.load('./variables/label_set/set_%s.npy' % label_type)

##################################################
# model parameter
num_classes = train_labels.shape[1]  # 标签的
num_filters = 512  # 卷积核的数量
num_hidden = 1000  # 隐藏层的数量
batch_size = 256
num_epochs = 1
dropout_rate = 0.2

###################################################
# Model
# single filter cnn (simple textcnn)
input = Input(shape=[train_fact_pad_seq.shape[1]])
embedding_layer = Embedding(input_dim=num_words + 1,
                            input_length=maxlen,
                            output_dim=256,
                            mask_zero=0,
                            name='Embedding')(input)
conv = Conv1D(num_filters, 3, strides=1, padding='same')(embedding_layer)
relu = Activation(activation='relu')(conv)
maxpool = GlobalMaxPooling1D()(relu)
bn = BatchNormalization()(maxpool)
drop1 = Dropout(dropout_rate)(bn)
dense = Dense(num_hidden, activation='relu')(drop1)
drop2 = Dropout(dropout_rate)(dense)
output = Dense(num_classes, activation='sigmoid')(drop2)
model = Model(inputs=input, outputs=output)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
plot_model(model, to_file="./pics/cnn_filter3.png",show_shapes=True)

##################################################
for epoch in range(num_epochs):
    model.fit(x=train_fact_pad_seq, y=train_labels,
              batch_size=batch_size, epochs=1,
              validation_data=(valid_fact_pad_seq, valid_labels), verbose=1)

    # 计算准确率 calculate accuracy
    predictions_valid = model.predict(valid_fact_pad_seq[:])
    # predictions_test = model.predict(test_fact_pad_seq[:])

    # 用validation集来验证
    predictions = predictions_valid

    sets = set_labels

    y1 = label2tag(valid_labels[:], sets)  # 将验证集标签由one-hot转为原标签
    y2 = predict2toptag(predictions, sets)  # 只取最高的
    y3 = predict2half(predictions, sets)  # 只取概率大于0.5的
    y4 = predict2tag(predictions, sets)  # y2与y3的交集，即有大于0.5的取大于0.5的，没有就取概率最高的

    # 只取最高置信度的准确率
    s1 = [str(y1[i]) == str(y2[i]) for i in range(len(y1))]
    print(sum(s1) / len(s1))
    # 只取置信度大于0.5的准确率
    s2 = [str(y1[i]) == str(y3[i]) for i in range(len(y1))]
    print(sum(s2) / len(s2))
    # 结合前两个
    s3 = [str(y1[i]) == str(y4[i]) for i in range(len(y1))]
    accuracy = int(np.round(sum(s3) / len(s3), 3) * 100)
    print(accuracy)

    # 计算f1 score calculate f1 score
    predictions_one_hot = predict1hot(predictions)
    f1_micro = f1_score(valid_labels, predictions_one_hot, average='micro')
    print('f1_micro_accusation:', f1_micro)
    f1_marco = f1_score(valid_labels, predictions_one_hot, average='macro')
    print('f1_macro_accusation:', f1_marco)
    # 取两者平均
    f1_average = int(np.round((f1_marco + f1_micro) / 2, 2) * 100)
    print('total:', f1_average)

    # save model
    model.save('./model/cnn_%s_token_%s_pad_%s_filter_%s_hidden_%s_epoch_%s_accu_%s_f1_%s.h5' % (
        label_type, num_words, maxlen, num_filters, num_hidden, epoch + 1, accuracy, f1_average))

    # excel 保存原label与prediction的对比
    r = pd.DataFrame({'label': y1, 'predict': y4})
    r.to_excel('./results/cnn_%s_token_%s_pad_%s_filter_%s_hidden_%s_epoch_%s_accu_%s_f1_%s.xlsx' % (
        label_type, num_words, maxlen, num_filters, num_hidden, epoch + 1, accuracy, f1_average),
               sheet_name='1', index=False)

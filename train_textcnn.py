import tensorflow as tf
import numpy as np

embedding_dim = 256  # embedding layer dimension is fixed to 256
num_words = 40000
maxlen = 400
label_type = 'accusation'

train_pad_seg = np.load('./variables/pad_sequences/train_pad_%d_%d.npy' % (maxlen, num_words))
valid_pad_seg = np.load('./variables/pad_sequences/valid_pad_%d_%d.npy' % (maxlen, num_words))
test_pad_seg = np.load('./variables/pad_sequences/test_pad_%d_%d.npy' % (maxlen, num_words))

y_train = np.load('./variables/labels/train_one_hot_%s.npy' % (label_type))
y_valid = np.load('./variables/labels/valid_one_hot_%s.npy' % (label_type))
y_test = np.load('./variables/labels/test_one_hot_%s.npy' % (label_type))


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


##################################################
# reset_graph()

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of character embedding (default: 256)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5,6", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 256, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob_train", 0.2, "Dropout keep probability for training(default: 0.5)")
tf.flags.DEFINE_float("dropout_keep_prob_test", 1.0, "Dropout keep probability for test(no dropout)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

FLAGS = tf.flags.FLAGS

##################################################
sequence_length = train_pad_seg.shape[1]
num_classes = y_train.shape[1]
vocab_size = num_words + 1
embedding_size = FLAGS.embedding_dim
filter_sizes = list(map(int, FLAGS.filter_sizes.split(",")))
num_filters = FLAGS.num_filters
l2_reg_lambda = FLAGS.l2_reg_lambda

# Placeholders for input, output and dropout
input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

# Keeping track of l2 regularization loss (optional)
l2_loss = tf.constant(0.0)

# Embedding layer
with tf.name_scope("embedding"):
    W = tf.Variable(
        tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
        name="W")
    embedded_chars = tf.nn.embedding_lookup(W, input_x)
    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

# Create a convolution + maxpool layer for each filter size
pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    with tf.name_scope("conv-maxpool-%s" % filter_size):
        # Convolution Layer
        filter_shape = [filter_size, embedding_size, 1, num_filters]  # num_filters卷积核的数量
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")  # 初始化权重
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        conv = tf.nn.conv2d(
            embedded_chars_expanded,  # 输入层的维度
            W,
            strides=[1, 1, 1, 1],  # 滑动步长
            padding="VALID",  # 图像边沿填充的方式
            name="conv")
        # Apply nonlinearity
        h = tf.nn.selu(tf.nn.bias_add(conv, b), name="selu")
        # Maxpooling over the outputs
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, sequence_length - filter_size + 1, 1, 1],  # 高为sequence_length - filter_size + 1，宽为1（因为全部加总）
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        pooled_outputs.append(pooled)

# Combine all the pooled features
num_filters_total = num_filters * len(filter_sizes)
h_pool = tf.concat(pooled_outputs, 3)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

# Add dropout
with tf.name_scope("dropout"):
    h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

# Final (unnormalized) scores and predictions
with tf.name_scope("output"):
    W = tf.get_variable(
        "W",
        shape=[num_filters_total, num_classes],
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
    l2_loss += tf.nn.l2_loss(W)
    l2_loss += tf.nn.l2_loss(b)
    scores = tf.nn.xw_plus_b(h_pool_flat, W, b, name="scores")
    # self.predictions = tf.argmax(self.scores, 1, name="predictions")
    predictions = tf.round(tf.nn.sigmoid(scores), name="predictions")  # 用sigmoid函数生成每个预测标签

# Calculate mean cross-entropy loss
with tf.name_scope("loss"):
    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=scores, labels=input_y)
    loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

learning_rate = 1e-3
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

# Accuracy
with tf.name_scope("accuracy"):
    correct_predictions = tf.equal(predictions, tf.round(input_y))
    accuracy = tf.reduce_mean(tf.reduce_min(tf.cast(correct_predictions, tf.float32), 1), name="accuracy")

########################################################################################################
# train / test parameter
n_epochs = 10
train_batch_size = 256
test_batch_size = 256
valid_batch_size = 256
train_n_batches = int(np.ceil(len(train_pad_seg) / train_batch_size))
test_n_batches = int(np.ceil(len(test_pad_seg) / test_batch_size))
valid_n_batches = int(np.ceil(len(valid_pad_seg) / valid_batch_size))


def fetch_batch(epoch, data_set, y, batch_index, n_batches, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(len(data_set), size=batch_size)
    X_batch = data_set[indices]
    y_batch = y[indices]
    return X_batch, y_batch


init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    # saver.restore(sess, "./textcnn/models/models_cnn%s/model_cnn_%s.ckpt" % ((np.max(valid_score),) * 2))

    test_score = []
    valid_score = []
    for epoch in range(n_epochs):
        train_batch_acc = []
        test_batch_acc = []
        valid_batch_acc = []
        for batch_index in range(train_n_batches):  # 但//取的是结果的最小整数，而/取得是实际的除法结果
            X_batch, y_batch = fetch_batch(epoch, train_pad_seg, y_train, batch_index, train_n_batches,
                                           train_batch_size)
            sess.run(train_op, feed_dict={
                input_x: X_batch,
                input_y: y_batch,
                dropout_keep_prob: FLAGS.dropout_keep_prob_train
            })
            acc_train = accuracy.eval(feed_dict={
                input_x: X_batch,
                input_y: y_batch,
                dropout_keep_prob: FLAGS.dropout_keep_prob_train
            })
            train_batch_acc.append(acc_train)
            print(epoch, 'batch:', batch_index, "Train accuracy:", acc_train)
        print(epoch, "Train accuracy:", np.mean(train_batch_acc))

        for batch_index in range(test_n_batches):
            X_test_batch, y_test_batch = fetch_batch(epoch, test_pad_seg, y_test, batch_index, test_n_batches,
                                                     test_batch_size)
            acc_test = accuracy.eval(feed_dict={
                input_x: X_test_batch,
                input_y: y_test_batch,
                dropout_keep_prob: FLAGS.dropout_keep_prob_test
            })
            test_batch_acc.append(acc_test)
        mean_test_score = np.mean(test_batch_acc)
        print(epoch, "Test accuracy:", mean_test_score)
        test_score.append(mean_test_score)

        for batch_index in range(valid_n_batches):
            X_valid_batch, y_valid_batch = fetch_batch(epoch, valid_pad_seg, y_valid, batch_index, valid_n_batches,
                                                       valid_batch_size)
            acc_val = accuracy.eval(feed_dict={
                input_x: X_valid_batch,
                input_y: y_valid_batch,
                dropout_keep_prob: FLAGS.dropout_keep_prob_test
            })
            valid_batch_acc.append(acc_val)
        mean_valid_score = np.mean(valid_batch_acc)
        print(epoch, "valid accuracy:", np.mean(valid_batch_acc))
        valid_score.append(mean_valid_score)

    save_path = saver.save(sess, "./model/models_textcnn%s/model_textcnn_%s.ckpt" % (
        np.max(valid_score), np.max(valid_score)))

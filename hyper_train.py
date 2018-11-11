from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import re
import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical

from hyper_model import Model
from data_helper import load_data, DataReader

flags = tf.flags

# data
flags.DEFINE_string('data_dir', 'data/', 'data directory. Should contain train.txt/valid.txt/test.txt with input data')
flags.DEFINE_string('load_model', None,
                    '(optional) filename of the model to load. Useful for re-starting training from a checkpoint')

# model params
flags.DEFINE_integer('rnn_size', 250, 'size of LSTM internal state')
flags.DEFINE_integer('highway_layers', 0, 'number of highway layers')
flags.DEFINE_integer('char_embed_size', 25, 'dimensionality of character embeddings')
flags.DEFINE_integer('hyper_input_size',7,'dimensionality of hypernetworks input')
flags.DEFINE_string('kernels', '[1,2,3,4,5,6,7]', 'CNN kernel widths')
flags.DEFINE_string('kernel_features', '[50,100,150,200,200,200,200]', 'number of features in the CNN kernel')
flags.DEFINE_integer('rnn_layers', 2, 'number of layers in the LSTM')
flags.DEFINE_float('dropout', 0.5, 'dropout. 0 = no dropout')
flags.DEFINE_float('recurrent_dropout', 0.9, 'recurrent dropout')
flags.DEFINE_integer('num_classes',53,'25 classed in sum ,but eos if another class')

# optimization
flags.DEFINE_float('learning_rate', 0.0001, 'starting learning rate')
flags.DEFINE_float('learning_rate_decay', 1, 'learning rate decay')
flags.DEFINE_float('decay_when', 1.0, 'decay if validation perplexity does not improve by more than this much')
flags.DEFINE_float('param_init', 0.05, 'initialize parameters at')
flags.DEFINE_integer('num_unroll_steps', 39, 'number of timesteps to unroll for')
flags.DEFINE_integer('batch_size', 20, 'number of sequences to train on in parallel')
flags.DEFINE_integer('max_epochs', 100, 'number of full passes through the training data')
flags.DEFINE_float('max_grad_norm', 5.0, 'normalize gradients at')
flags.DEFINE_integer('max_word_length', 35, 'maximum word length')

# bookkeeping
flags.DEFINE_integer('seed', 3435, 'random number generator seed')
flags.DEFINE_integer('print_every', 20, 'how often to print current loss')
flags.DEFINE_string('EOS', '+',
                    '<EOS> symbol. should be a single unused character (like +) for PTB and blank for others')

FLAGS = flags.FLAGS


def main():
    word_vocab, char_vocab, label_vocab, word_tensors, char_tensors, label_tensors, mask_tensors = \
        load_data(FLAGS.data_dir, FLAGS.num_unroll_steps, FLAGS.max_word_length)
#word_tensor, char_tensor,label_tensor, batch_size, num_class, num_unroll_steps
    train_reader = DataReader(word_tensors['train'], char_tensors['train'],label_tensors['train'],mask_tensors['train'],
                              FLAGS.batch_size, FLAGS.num_classes)
    test_reader = DataReader(word_tensors['test'], char_tensors['test'],label_tensors['test'],mask_tensors['test'],
                             FLAGS.batch_size, FLAGS.num_classes)
    dev_reader = DataReader(word_tensors['dev'], char_tensors['dev'], label_tensors['dev'],mask_tensors['dev'],
                             FLAGS.batch_size, FLAGS.num_classes)
    domainptb_reader = DataReader(word_tensors['domain_ptb'], char_tensors['domain_ptb'],label_tensors['domain_ptb'],mask_tensors['domain_ptb'],
                             FLAGS.batch_size, FLAGS.num_classes)

    print('initialized all dataset readers')

    args = FLAGS
    args.char_vocab_size = char_vocab.size
    args.word_vocab_size = word_vocab.size
    args.word_vocab = word_vocab


    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(FLAGS.seed)
        np.random.seed(seed=FLAGS.seed)

        ''' build training graph '''
        # para_init=0.05
        initializer = tf.random_uniform_initializer(-FLAGS.param_init, FLAGS.param_init)
        with tf.variable_scope("Model", initializer=initializer):
            train_model = Model(args)

    with tf.Session(graph=g) as session:
        tf.initialize_all_variables().run()
        print('Created and initialized fresh model.')


        ''' training starts here '''

        model_path = "./hyper_model.ckpt"
        saver = tf.train.Saver()
        valid_best_accuracy = 0

        # training start
        for epoch in range(5):
            domainptb_iter = domainptb_reader.iter()

            for i_train in range(100):
                print('****************  TWITTER ***********************')
                avg_train_loss = []
                avg_train_correct = []
                avg_train_sum = []
                for x, y, z, m in train_reader.iter():
                    sentence_length = []
                    for batch in m:
                        sentence_length.append(sum(batch))
                    loss, correct_num, sum_num, _ = session.run([
                        # sequence order change
                        train_model.class_loss,
                        train_model.correct_num,
                        train_model.sum_num,
                        train_model.class_train_op
                    ], {
                        train_model.input_: x,
                        train_model.class_targets: y,
                        train_model.input_word: z,
                        train_model.input_mask: m,
                        train_model.sentence_length: sentence_length,
                        train_model.dropout: 0.5,
                        train_model.learning_rate: FLAGS.learning_rate * (FLAGS.learning_rate_decay ** epoch),
                        train_model.recurrent_dropout: FLAGS.recurrent_dropout
                    })
                    avg_train_loss.append(loss)
                    avg_train_correct.append(correct_num)
                    avg_train_sum.append(sum_num)
                print(
                    'the %d training epoch of average loss is:%f' % (i_train, sum(avg_train_loss) / len(avg_train_loss)))
                print('the class accuracy of %d !!!!!!!!!!!!!training epoch is %f:' % (i_train, sum(avg_train_correct) / sum(avg_train_sum)))


                print('****************  PTB  ***********************')
                avg_ptb_loss = []
                avg_ptb_correct = []
                avg_ptb_sum = []
                for i in range(30):
                    x,y,z,m = next(domainptb_iter)

                    sentence_length = []
                    for batch in m:
                        sentence_length.append(sum(batch))

                    loss, correct_num, sum_num, _ = session.run([
                        # sequence order change
                        train_model.class_loss,
                        train_model.correct_num,
                        train_model.sum_num,
                        train_model.class_train_op
                    ], {
                        train_model.input_: x,
                        train_model.class_targets: y,
                        train_model.input_word: z,
                        train_model.input_mask: m,
                        train_model.sentence_length: sentence_length,
                        train_model.dropout: 0.5,
                        train_model.learning_rate: FLAGS.learning_rate * (FLAGS.learning_rate_decay ** epoch),
                        train_model.recurrent_dropout: FLAGS.recurrent_dropout
                    })
                    avg_ptb_loss.append(loss)
                    avg_ptb_correct.append(correct_num)
                    avg_ptb_sum.append(sum_num)
                print(
                    'the %d training epoch of average loss is:%f' % (i_train, sum(avg_ptb_loss) / len(avg_ptb_loss)))
                print('the class accuracy of %d !!!!!!!!!!!!!training epoch is %f:' % (
                    i_train, sum(avg_ptb_correct) / sum(avg_ptb_sum)))


                print('**************** NOW VALID ***********************')
                avg_dev_loss = []
                file = open('dev_text.txt','w')
                error_num=0
                for x, y, z, m in dev_reader.iter():
                    sentence_length = []
                    for batch in m:
                        sentence_length.append(sum(batch))
                    loss, y_pred, train_index, test_index, input_hyper = session.run([
                        train_model.class_loss,
                        train_model.y_pred1,
                        train_model.train_index,
                        train_model.test_index,
                        train_model.input_hyper
                    ], {
                        train_model.input_: x,
                        train_model.class_targets: y,
                        train_model.input_word: z,
                        train_model.input_mask: m,
                        train_model.sentence_length: sentence_length,
                        train_model.dropout: 1,
                        train_model.learning_rate: FLAGS.learning_rate * (FLAGS.learning_rate_decay ** epoch),
                        train_model.recurrent_dropout: 1
                    })

                    word_batch = z.flatten()

                    # substitute the #@RTURL
                    patternHT = '#[\w]+'
                    patternUSR = '@[\w]+'
                    patternURL = 'http|www\.|^com[^\w]'

                    inputword = np.ndarray.flatten(z)
                    # inputword[batch_size, time]
                    y_predtrue = []
                    # y_pred[batch_size * time, num_classes]
                    for label, word in zip(y_pred, inputword):
                        if re.match(patternHT, word_vocab._index2token[word]):
                            y_predtrue.append(
                                to_categorical([label_vocab._token2index['HT']], num_classes=args.num_classes)[0])
                        elif re.match(patternURL, args.word_vocab._index2token[word]):
                            y_predtrue.append(
                                to_categorical([label_vocab._token2index['URL']], num_classes=args.num_classes)[0])
                        elif re.match(patternUSR, args.word_vocab._index2token[word]):
                            y_predtrue.append(
                                to_categorical([label_vocab._token2index['USR']], num_classes=args.num_classes)[0])
                        else:
                            label_index = np.argmax(label)
                            ht_index = label_vocab._token2index['HT']
                            url_index = label_vocab._token2index['URL']
                            usr_index = label_vocab._token2index['USR']
                            while (label_index == ht_index) or (label_index == url_index) or (label_index == usr_index):
                                label[label_index] = 0
                                label_index = np.argmax(label)
                            y_predtrue.append(label)
                    y = np.reshape(y,(-1,args.num_classes))

                    class_correct_prediction = np.equal(np.argmax(y_predtrue, 1), np.argmax(y, 1))

                    for index,y_p,y,word in zip(class_correct_prediction,train_index,test_index,word_batch):
                        if index==False:
                            a=label_vocab._index2token[y_p]
                            b=label_vocab._index2token[y]
                            c=word_vocab._index2token[word]
                            if label_vocab._index2token[y]!='+':
                                error_num+=1

                                try:
                                    file.write(c + '\t' + b + '\t' + a)
                                    file.write('\n')
                                except Exception as e:
                                    print(e)
                                    pass
                    avg_dev_loss.append(loss)
                file.close()

                print('the %d validation epoch of average loss is:%f' % (i_train, sum(avg_dev_loss) / len(avg_dev_loss)))
                print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@the true accuracy of %d validatoin epoch is %f:' % (
                    i_train, (2242 - error_num) / 2242))

                target_valid_accuracy = (2242 - error_num) / 2242
                if target_valid_accuracy > valid_best_accuracy:
                    #save_path = saver.save(session, model_path)
                    valid_best_accuracy = target_valid_accuracy
                    print("^^^^^^^^^^  SAVE MODEL !!!  ^^^^^^^^^^^^")
                else:
                    print('^^^^^^^^^^  NOT SAVED !!!  ^^^^^^^^^^^^^')


                #TESTING
                print('**************** NOW TESTING ***********************')

                avg_test_loss = []
                file = open('wrong_text.txt','w')
                error_num = 0
                for x, y, z, m in test_reader.iter():
                    sentence_length = []
                    for batch in m:
                        sentence_length.append(sum(batch))
                    loss, y_pred,train_index,test_index= session.run([
                        train_model.class_loss,
                        train_model.y_pred1,
                        train_model.train_index,
                        train_model.test_index
                    ], {
                        train_model.input_: x,
                        train_model.class_targets: y,
                        train_model.input_word: z,
                        train_model.input_mask: m,
                        train_model.sentence_length: sentence_length,
                        train_model.dropout: 1,
                        train_model.learning_rate: FLAGS.learning_rate * (FLAGS.learning_rate_decay ** epoch),
                        train_model.recurrent_dropout: 1
                    })
                    word_batch = z.flatten()

                    # substitute the #@RTURL
                    patternHT = '#[\w]+'
                    patternUSR = '@[\w]+'
                    patternURL = 'http|www\.|^com[^\w]'

                    inputword = np.ndarray.flatten(z)
                    y_predtrue = []
                    for label, word in zip(y_pred, inputword):
                        if re.match(patternHT, word_vocab._index2token[word]):
                            y_predtrue.append(
                                to_categorical([label_vocab._token2index['HT']], num_classes=args.num_classes)[0])
                        elif re.match(patternURL, args.word_vocab._index2token[word]):
                            y_predtrue.append(
                                to_categorical([label_vocab._token2index['URL']], num_classes=args.num_classes)[0])
                        elif re.match(patternUSR, args.word_vocab._index2token[word]):
                            y_predtrue.append(
                                to_categorical([label_vocab._token2index['USR']], num_classes=args.num_classes)[0])
                        else:
                            label_index = np.argmax(label)
                            ht_index = label_vocab._token2index['HT']
                            url_index = label_vocab._token2index['URL']
                            usr_index = label_vocab._token2index['USR']
                            while (label_index == ht_index) or (label_index == url_index) or (label_index == usr_index):
                                label[label_index] = 0
                                label_index = np.argmax(label)
                            y_predtrue.append(label)
                    y = np.reshape(y, (-1, args.num_classes))

                    class_correct_prediction = np.equal(np.argmax(y_predtrue, 1), np.argmax(y, 1))

                    for index,y_p,y,word in zip(class_correct_prediction,train_index,test_index,word_batch):
                        if index==False:
                            a=label_vocab._index2token[y_p]
                            b=label_vocab._index2token[y]
                            c=word_vocab._index2token[word]
                            if label_vocab._index2token[y]!='+':
                                error_num+=1

                                try:
                                    file.write(c + '\t' + b + '\t' + a)
                                    file.write('\n')
                                except Exception as e:
                                    print(e)
                                    pass



                    avg_test_loss.append(loss)

                file.close()
                print('the %d testing epoch of average loss is:%f' % (i_train, sum(avg_test_loss) / len(avg_test_loss)))
                print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@the true accuracy of %d testing epoch is %f:' % (i_train, (2291-error_num) / 2291))

        saver = tf.train.Saver()
        load_path = saver.restore(session, model_path)
        print('######################## visualization ###############################')
        ptb_vision = open('ptb_vision', 'w')
        for x, y, z, m in domainptb_reader.iter():
            sentence_length = []
            for batch in m:
                sentence_length.append(sum(batch))
            loss, correct_num, sum_num, input_hyper = session.run([
                # sequence order change
                train_model.class_loss,
                train_model.correct_num,
                train_model.sum_num,
                train_model.input_hyper
            ], {
                train_model.input_: x,
                train_model.class_targets: y,
                train_model.input_word: z,
                train_model.input_mask: m,
                train_model.sentence_length: sentence_length,
                train_model.dropout: 1,
                train_model.learning_rate: FLAGS.learning_rate,
                train_model.recurrent_dropout: 1
            })
            input_hyper = input_hyper.tolist()
            z = z.tolist()
            for num_sentence, embedding in zip(z, input_hyper):
                word_sentence = [word_vocab._index2token[i] for i in num_sentence]
                ptb_vision.write('*'.join(word_sentence) + '\t')
                for prob in embedding:
                    prob = [str(i) for i in prob]
                    ptb_vision.write(' '.join(prob) + '*')
                ptb_vision.write('\n')
        ptb_vision.close()


        fine_tune = open('train_vision', 'w')
        for x, y, z, m in train_reader.iter():
            sentence_length = []
            for batch in m:
                sentence_length.append(sum(batch))
            loss, correct_num, sum_num, input_hyper = session.run([
                # sequence order change
                train_model.class_loss,
                train_model.correct_num,
                train_model.sum_num,
                train_model.input_hyper
            ], {
                train_model.input_: x,
                train_model.class_targets: y,
                train_model.input_word: z,
                train_model.input_mask: m,
                train_model.sentence_length: sentence_length,
                train_model.dropout: 1,
                train_model.learning_rate: FLAGS.learning_rate,
                train_model.recurrent_dropout: 1
            })
            input_hyper = input_hyper.tolist()
            z = z.tolist()
            for num_sentence, embedding in zip(z, input_hyper):
                word_sentence = [word_vocab._index2token[i] for i in num_sentence]
                fine_tune.write('*'.join(word_sentence) + '\t')
                for prob in embedding:
                    prob = [str(i) for i in prob]
                    fine_tune.write(' '.join(prob) + '*')
                fine_tune.write('\n')
        fine_tune.close()

        dev_vision = open('dev_vision', 'w')
        for x, y, z, m in dev_reader.iter():
            sentence_length = []
            for batch in m:
                sentence_length.append(sum(batch))
            loss, y_pred, train_index, test_index, input_hyper = session.run([
                train_model.class_loss,
                train_model.y_pred1,
                train_model.train_index,
                train_model.test_index,
                train_model.input_hyper
            ], {
                train_model.input_: x,
                train_model.class_targets: y,
                train_model.input_word: z,
                train_model.input_mask: m,
                train_model.sentence_length: sentence_length,
                train_model.dropout: 1,
                train_model.recurrent_dropout: 1
            })
            input_hyper = input_hyper.tolist()
            z = z.tolist()
            for num_sentence, embedding in zip(z, input_hyper):
                word_sentence = [word_vocab._index2token[i] for i in num_sentence]
                dev_vision.write('*'.join(word_sentence) + '\t')
                for prob in embedding:
                    prob = [str(i) for i in prob]
                    dev_vision.write(' '.join(prob) + '*')
                dev_vision.write('\n')
        dev_vision.close()

        test_vision = open('test_vision', 'w')
        for x, y, z, m in test_reader.iter():
            sentence_length = []
            for batch in m:
                sentence_length.append(sum(batch))
            loss, y_pred, train_index, test_index, input_hyper = session.run([
                train_model.class_loss,
                train_model.y_pred1,
                train_model.train_index,
                train_model.test_index,
                train_model.input_hyper
            ], {
                train_model.input_: x,
                train_model.class_targets: y,
                train_model.input_word: z,
                train_model.input_mask: m,
                train_model.sentence_length: sentence_length,
                train_model.dropout: 1,
                train_model.recurrent_dropout: 1
            })
            input_hyper = input_hyper.tolist()
            z = z.tolist()
            for num_sentence, embedding in zip(z, input_hyper):
                word_sentence = [word_vocab._index2token[i] for i in num_sentence]
                test_vision.write('*'.join(word_sentence) + '\t')
                for prob in embedding:
                    prob = [str(i) for i in prob]
                    test_vision.write(' '.join(prob) + '*')
                test_vision.write('\n')
        test_vision.close()




if __name__ == "__main__":
    start = time.time()
    main()
    wholetime = time.time() - start
    hour = int(wholetime) // 3600
    minute = int(wholetime) % 60
    sec = wholetime - 3600 * hour - 60 * minute
    print("whole time:\n")
    print("%d:%d:%f" % (hour, minute, sec))

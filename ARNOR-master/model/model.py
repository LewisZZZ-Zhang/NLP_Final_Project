import tensorflow as tf
from tqdm import tqdm
import numpy as np




class model(object):
    def __init__(self, config, mode, trustable_pattern=None):
        self.config = config
        self.lr = config['lr'] if 'lr' in config else None
        self.id_to_label = {v: k for k, v in self.config['label_dict'].items()}
        self.trustable_pattern = trustable_pattern
        self.mode = mode.lower()

    def placeholder_op(self):
        self.word_ids = tf.compat.v1.placeholder(name='word_ids',
                                       shape=[None, None],
                                       dtype=tf.int32)

        self.position1_ids = tf.compat.v1.placeholder(name='position1_ids', 
                                            shape=[None, None],
                                            dtype=tf.int32)

        self.position2_ids = tf.compat.v1.placeholder(name='position2_ids',
                                            shape=[None, None],
                                            dtype=tf.int32)

        self.type_ids = tf.compat.v1.placeholder(name='type_ids',
                                       shape=[None, None],
                                       dtype=tf.int32)

        self.att_labels = tf.compat.v1.placeholder(name='att_label',
                                        shape = [None, None],
                                        dtype=tf.float32)
        
        self.lengths = tf.compat.v1.placeholder(name='lengths',
                                     shape=[None],
                                     dtype=tf.int32)

        self.label_ids = tf.compat.v1.placeholder(name='label_ids',
                                        shape=[None],
                                        dtype=tf.int32)

        self.learn_rate = tf.compat.v1.placeholder(name='lr',
                                         shape=[],
                                         dtype=tf.float32)

        self.dropout = tf.compat.v1.placeholder(name='dropout',
                                      shape=[],
                                      dtype=tf.float32
                                      )




    def forward(self):
        
        with tf.compat.v1.variable_scope('embedding'):
            word_embedding_matrix = tf.compat.v1.get_variable(name='word_embedding_matrix',
                                                    shape=[self.config['word_num'], self.config['word_dim']],
                                                    dtype=tf.float32,
                                                    initializer=tf.keras.initializers.GlorotUniform()
                                                    )

            pos1_embedding_matrix = tf.compat.v1.get_variable(name='pos1_embedding_matrix',
                                                    shape=[self.config['position_num'], self.config['position_dim']],
                                                    dtype=tf.float32,
                                                    initializer=tf.keras.initializers.GlorotUniform()
                                                    )

            pos2_embedding_matrix = tf.compat.v1.get_variable(name='pos2_embedding_matrix',
                                                    shape=[self.config['position_num'], self.config['position_dim']],
                                                    dtype=tf.float32,
                                                    initializer=tf.keras.initializers.GlorotUniform()
                                                    )

            type_embedding_matrix = tf.compat.v1.get_variable(name='type_embedding_matrix',
                                                    shape=[self.config['type_num'], self.config['type_dim']],
                                                    dtype=tf.float32,
                                                    initializer=tf.keras.initializers.GlorotUniform()
                                                    )

            word_embeddings = tf.nn.embedding_lookup(params=word_embedding_matrix,
                                                     ids=self.word_ids,
                                                     name='word_embeddings_look_up')

            pos1_embeddings = tf.nn.embedding_lookup(params=pos1_embedding_matrix,
                                                     ids=self.position1_ids,
                                                     name='pos1_embeddings_look_up')

            pos2_embeddings = tf.nn.embedding_lookup(params=pos2_embedding_matrix,
                                                     ids=self.position2_ids,
                                                     name='pos2_embeddings_look_up')

            type_embeddings = tf.nn.embedding_lookup(params=type_embedding_matrix,
                                                     ids=self.type_ids,
                                                     name='type_embeddings_look_up')

            bilstm_input = tf.concat([word_embeddings, pos1_embeddings, pos2_embeddings, type_embeddings], -1)
            # tf.nn.dropout in TF2 uses `rate` (1 - keep_prob)
            bilstm_input = tf.nn.dropout(bilstm_input, rate=1.0 - self.dropout)

        with tf.compat.v1.variable_scope('BiLSTM_ATT'):
            # Use Keras Bidirectional LSTM for TF2/Keras3 compatibility
            max_len = tf.shape(bilstm_input)[1]
            mask = tf.sequence_mask(self.lengths, maxlen=max_len)
            # Give explicit, stable names to Keras layers so checkpoints restore
            bilstm_layer = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(self.config['hidden_dim']//2, return_sequences=True, name='lstm_cell'),
                merge_mode='concat', name='bilstm')
            # (B, max_length, hidden_dim)
            hiddens = bilstm_layer(bilstm_input, mask=mask)

            # attention layer
            # (B, max_length, hidden_dim)
            hiddens_for_att =  tf.squeeze(tf.keras.layers.Dense(1, activation=tf.tanh, use_bias=False, name='att_dense')(hiddens), -1)
            # (B, max_length)
            self.attentions = self.attention_fun(hiddens_for_att, self.lengths)
            # (B, hidden_dim)
            hidden_att = tf.reduce_sum(hiddens * tf.expand_dims(self.attentions, -1), 1)
            hidden_att = tf.tanh(hidden_att)
            # FC
            #hidden_att = tf.keras.layers.Dense(self.config['hidden_dim'])(hidden_att)
            self.logits = tf.keras.layers.Dense(self.config['label_num'], name='fc_logits')(hidden_att)



    @staticmethod
    def attention_fun(hiddens, lengths):
        batchsize = tf.shape(hiddens)[0]
        max_length = tf.reduce_max(lengths)

        def __attention__(bid):
            # (1)
            length = lengths[bid]
            # (length)
            attention_logit = hiddens[bid][:length]
            # (length)
            attention = tf.nn.softmax(attention_logit)
            # (max_length), (a_1,a_2...a_length,0,0,0...)
            return tf.concat([attention, tf.zeros(max_length-length)], -1)

        # [(max_length), (max_length)...] -> [batchsize, max_length]
        return tf.concat(tf.map_fn( __attention__, tf.range(batchsize), dtype=tf.float32), 0)



    def loss_op(self):

        with tf.compat.v1.variable_scope('loss'):
            loss_c = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ids, logits=self.logits)

            if self.config['attention_regularization']:
                self.kl = tf.keras.losses.kullback_leibler_divergence(y_true=self.att_labels, y_pred=self.attentions)
                attentions = tf.clip_by_value(self.attentions, 1e-10, 1.0)
                loss_a = -tf.reduce_sum(self.att_labels * tf.math.log(attentions), -1)
                self.loss = tf.reduce_mean(loss_c + self.config['beta'] * loss_a)
            else:
                self.loss = tf.reduce_mean(loss_c)


    def add_train_op(self):
        """Defines self.train_op that performs an update on a batch
        Args:
            lr_method: (string) sgd method, for example "adam"
            lr: (tf.placeholder) tf.float32, learning rate
            loss: (tensor) tf.float32 loss to minimize
            clip: (python float) clipping of gradient. If < 0, no clipping
        """
        _lr_m = self.config['lr_method'].lower() # lower to make sure
        lr = self.learn_rate
        clip = self.config['clip']
        loss = self.loss
        with tf.compat.v1.variable_scope("train_step"):
            if _lr_m == 'adam': # sgd method
                optimizer = tf.compat.v1.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.compat.v1.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.compat.v1.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.compat.v1.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            if clip > 0: # gradient clipping if clip is positive
                grads, vs = zip(*optimizer.compute_gradients(loss))
                grads, gnorm = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)


    def predict(self, data_batch, mode=None):
        """Run forward and return predicted labels.

        Args:
            data_batch: list of features in the same format as `get_feed` expects.
            mode: optional override for mode used when building the feed (e.g. 'val').
        """
        feed = self.get_feed(data_batch, mode=mode)
        data_score = self.sess.run(self.logits, feed)
        data_pre = np.argmax(data_score, -1)
        return data_pre
        
        
    def Session(self, graph):
        self.sess = tf.compat.v1.Session(graph=graph)
        self.saver = tf.compat.v1.train.Saver()

        if self.mode.lower() == 'train' and not self.config['restore']:
            print('random init')
            self.sess.run(tf.compat.v1.global_variables_initializer())
        elif self.mode.lower() == 'train':
            print('restore model from ckpt_model for train')
            # restore from the latest checkpoint file in the directory
            ckpt = tf.train.latest_checkpoint('./ckpt_model/')
            if ckpt is None:
                raise FileNotFoundError('No checkpoint found in ./ckpt_model/')
            self.saver.restore(self.sess, ckpt)


    def build(self, graph):
        self.placeholder_op()
        self.forward()
        if self.mode.lower() == 'train':
            self.loss_op()
            self.add_train_op()
        self.Session(graph)


    def get_feed(self, datas, mode=None):
        """Build a feed dict for `sess.run`.

        Args:
            datas: list containing features. For training this should contain
                   7 items: features, position1s, position2s, typess, att_labels, lengths, labels.
                   For prediction/eval this should contain 5 items: features, position1s, position2s, typess, lengths.
            mode: optional string to override `self.mode` when constructing the feed.
        """
        effective_mode = mode if mode is not None else self.mode

        if len(datas) > 5:  # for train-like inputs
            assert len(datas) == 7, [len(datas)]
            features, position1s, position2s, typess, att_labels, lengths, labels = datas
        else:  # for predict/eval
            assert len(datas) == 5, [len(datas)]
            features, position1s, position2s, typess, lengths = datas
            att_labels, labels = None, None

        feed = {
            self.word_ids: features,
            self.position1_ids: position1s,
            self.position2_ids: position2s,
            self.type_ids: typess,
            self.lengths: lengths,
        }

        if effective_mode == 'train':
            feed[self.dropout] = self.config['dropout']
            feed[self.learn_rate] = self.lr
            feed[self.att_labels] = att_labels
            feed[self.label_ids] = labels
        else:
            # during eval/predict turn off dropout
            feed[self.dropout] = 1.0

        return feed


    def run_train(self, train_data):
        all_iter = len(train_data)//self.config['batchsize']
        if len(train_data) % self.config['batchsize']!=0:
            all_iter += 1

        train_data_iter = train_data.minibatch(self.config['batchsize'],
                                               shuffle=True,
                                               redistribution=self.config['redistribution'],
                                               trustable_pattern=self.trustable_pattern)

        pbar = tqdm(total=all_iter)

        attentions = []
        kls = []
        patternss = []
        labelss = []

        for patterns, features, position1s, position2s, typess, lengths, att_labels, labels in train_data_iter:
            feed = self.get_feed([features, position1s, position2s, typess, att_labels, lengths, labels])
            if self.config['attention_regularization']:
                _, loss_batch, attention, kl= self.sess.run([self.train_op, self.loss, self.attentions, self.kl], feed)
                attentions.extend(attention)
                kls.extend(kl)
            else:
                _, loss_batch = self.sess.run([self.train_op, self.loss], feed)
            patternss.extend(patterns)
            labelss.extend(labels)
            pbar.set_description('loss: {}'.format(loss_batch))
            pbar.update(1)
        pbar.close()
        # ensure checkpoint directory exists and use a stable filename prefix
        import os
        ckpt_dir = './ckpt_model'
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_prefix = os.path.join(ckpt_dir, 'model')
        self.saver.save(self.sess, ckpt_prefix)
        if self.config['attention_regularization']:
            return attentions, kls, patternss, labelss
        else:
            return patternss, labelss


    def run_evaule(self, val_data):
        # If this model was built as the training model (mode 'train'), its
        # variables are already initialized in the session after training.
        # In that case, skip attempting to restore from checkpoints and use
        # the in-memory weights directly for evaluation.
        # decide whether to attempt checkpoint restore (only for non-train modes)
        model_dir = None
        do_restore = (self.mode != 'train')
        if self.mode == 'train':
            # If this instance is the training model, variables are already in-memory
            # after training, so we skip attempting to restore from disk.
            print('Model built in train mode: using in-memory weights for evaluation.')
        else:
            model_dir = './ckpt_model/'
            model_dir = './best_model/' if self.mode == 'test' else model_dir
            print('restore model from {} for val or test'.format(model_dir))
        # Robust restore: only restore variables that exist in the checkpoint and in the current graph.
        if do_restore:
            ckpt = tf.train.latest_checkpoint(model_dir)
            if ckpt is None:
                raise FileNotFoundError('No checkpoint found in {}'.format(model_dir))

            # Attempt a safe partial restore: only restore variables that exist in both
            # the checkpoint and the current graph. If partial restore fails, skip
            # restoring to avoid crashing on incompatible Keras/TF2 checkpoint keys.
            ckpt_vars = {name: shape for name, shape in tf.train.list_variables(ckpt)}
            import re
            all_gvars = {v.name.split(':')[0]: v for v in tf.compat.v1.global_variables()}
            # Build a normalized mapping that strips Keras auto-increment suffixes
            # like `_1`, `_2` from layer tokens so we can match checkpoint names
            # saved from a model built in a different Keras UID state.
            normalized_gvars = {}
            for name, var in all_gvars.items():
                parts = name.split('/')
                norm_parts = [re.sub(r'_[0-9]+$', '', p) for p in parts]
                norm_name = '/'.join(norm_parts)
                # prefer exact name if present, otherwise keep first seen
                if norm_name not in normalized_gvars:
                    normalized_gvars[norm_name] = var

            restore_names = []
            restore_vars = {}
            for ck_name in ckpt_vars.keys():
                if ck_name in all_gvars:
                    restore_names.append(ck_name)
                    restore_vars[ck_name] = all_gvars[ck_name]
                elif ck_name in normalized_gvars:
                    # map checkpoint name to the normalized variable in current graph
                    restore_names.append(ck_name)
                    restore_vars[ck_name] = normalized_gvars[ck_name]

            if restore_vars:
                # Load and assign variables individually to avoid Saver name-matching
                # issues when Keras has appended numeric suffixes to layer names.
                restored = 0
                for ck_name, var in restore_vars.items():
                    try:
                        val = tf.train.load_variable(ckpt, ck_name)
                        self.sess.run(var.assign(val))
                        restored += 1
                    except Exception as e:
                        print(f'Failed to restore variable {ck_name}:', e)
                print(f'Partial restore: assigned {restored} variables from checkpoint.')
                if restored == 0:
                    print('No variables were restored from checkpoint. Continuing without restore.')
            else:
                print('No overlapping variables between checkpoint and graph. Skipping restore.')

        all_iter_val = len(val_data)//self.config['batchsize']
        if len(val_data) % self.config['batchsize']!=0:
            all_iter_val +=1

        val_data_iter = val_data.minibatch(self.config['batchsize'])
        predict_val_epoch = []
        label_val_epoch = []

        pbar = tqdm(total=all_iter_val)
        for patterns, features, position1s, position2s, typess, lengths, att_labels, labels in val_data_iter:
            # explicitly request prediction in 'val' mode so get_feed won't expect
            # training-only placeholders like `att_label`.
            predict_val_batch = self.predict([features, position1s, position2s, typess, lengths], mode='val')
            predict_val_epoch.extend(predict_val_batch)
            label_val_epoch.extend(labels)
            pbar.update(1)
        pbar.close()


        accs = np.mean([int(p==l) for p, l in zip(predict_val_epoch, label_val_epoch)])
        cur_score = {}

        for i in range(self.config['label_num']):
            i = self.id_to_label[i]
            pre_true_count = 0
            lab_count = 0
            pre_count = 0
            for pre, lab in zip(predict_val_epoch, label_val_epoch):
                pre = self.id_to_label[pre]
                lab = self.id_to_label[lab]
                if pre == i:
                    pre_count += 1
                if lab == i:
                    lab_count += 1
                if pre == i and lab == i:
                    pre_true_count += 1

            P = (pre_true_count / pre_count) if pre_count > 0 else 0
            R = (pre_true_count / lab_count) if lab_count > 0 else 0
            F = (2 * (P * R) / (P + R)) if (P + R) > 0 else 0
            cur_score[str(i)] = {'P': P, 'R': R, 'F1': F}

        pre_true_count = 0
        lab_count = 0
        pre_count = 0
        for pre, lab in zip(predict_val_epoch, label_val_epoch):
            pre = self.id_to_label[pre]
            lab = self.id_to_label[lab]

            if pre != 'None':
                pre_count += 1
            if lab != 'None':
                lab_count += 1
            if pre == lab and pre != 'None':
                pre_true_count += 1
        P = (pre_true_count / pre_count) if pre_count > 0 else 0
        R = (pre_true_count / lab_count) if lab_count > 0 else 0
        F = (2 * (P * R) / (P + R)) if (P + R) > 0 else 0
        cur_score['all'] = {'P': P, 'R': R, 'F1': F}

        return cur_score, accs


    def update_trustable_pattern(self, trustable_pattern ):
        self.trustable_pattern  =  trustable_pattern

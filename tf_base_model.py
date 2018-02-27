from __future__ import print_function
from collections import deque
from datetime import datetime
import logging
import os
import pprint as pp
import time

import numpy as np
import tensorflow as tf

from tf_utils import shape


class TFBaseModel(object):

    """Interface containing some boilerplate code for training tensorflow models.

    Subclassing models must implement self.calculate_loss(), which returns a tensor for the batch loss.
    Code for the training loop, parameter updates, checkpointing, and inference are implemented here and
    subclasses are mainly responsible for building the computational graph beginning with the placeholders
    and ending with the loss tensor.

    Args:
        reader: Class with attributes train_batch_generator, val_batch_generator, and test_batch_generator
            that yield dictionaries mapping tf.placeholder names (as strings) to batch data (numpy arrays).
        batch_size: Minibatch size.
        learning_rate: Learning rate.
        optimizer: 'rms' for RMSProp, 'adam' for Adam, 'sgd' for SGD
        grad_clip: Clip gradients elementwise to have norm at most equal to grad_clip.
        regularization_constant:  Regularization constant applied to all trainable parameters.
        keep_prob: 1 - p, where p is the dropout probability
        early_stopping_steps:  Number of steps to continue training after validation loss has
            stopped decreasing.
        warm_start_init_step:  If nonzero, model will resume training a restored model beginning
            at warm_start_init_step.
        num_restarts:  After validation loss plateaus, the best checkpoint will be restored and the
            learning rate will be halved.  This process will repeat num_restarts times.
        enable_parameter_averaging:  If true, model saves exponential weighted averages of parameters
            to separate checkpoint file.
        min_steps_to_checkpoint:  Model only saves after min_steps_to_checkpoint training steps
            have passed.
        log_interval:  Train and validation accuracies are logged every log_interval training steps.
        loss_averaging_window:  Train/validation losses are averaged over the last loss_averaging_window
            training steps.
        num_validation_batches:  Number of batches to be used in validation evaluation at each step.
        log_dir: Directory where logs are written.
        checkpoint_dir: Directory where checkpoints are saved.
        prediction_dir: Directory where predictions/outputs are saved.
    """

    def __init__(
        self,
        reader=None,
        batch_sizes=[128],
        num_training_steps=20000,
        learning_rates=[.01],
        beta1_decays=[.99],
        optimizer='adam',
        grad_clip=5,
        regularization_constant=0.0,
        keep_prob=1.0,
        patiences=[3000],
        warm_start_init_step=0,
        enable_parameter_averaging=False,
        min_steps_to_checkpoint=100,
        log_interval=20,
        logging_level=logging.INFO,
        loss_averaging_window=100,
        validation_batch_size=64,
        log_dir='logs',
        checkpoint_dir='checkpoints',
        prediction_dir='predictions',
    ):

        assert len(batch_sizes) == len(learning_rates) == len(patiences)
        self.batch_sizes = batch_sizes
        self.learning_rates = learning_rates
        self.beta1_decays = beta1_decays
        self.patiences = patiences
        self.num_restarts = len(batch_sizes) - 1
        self.restart_idx = 0
        self.update_train_params()

        self.reader = reader
        self.num_training_steps = num_training_steps
        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.regularization_constant = regularization_constant
        self.warm_start_init_step = warm_start_init_step
        self.keep_prob_scalar = keep_prob
        self.enable_parameter_averaging = enable_parameter_averaging
        self.min_steps_to_checkpoint = min_steps_to_checkpoint
        self.log_interval = log_interval
        self.loss_averaging_window = loss_averaging_window
        self.validation_batch_size = validation_batch_size

        self.log_dir = log_dir
        self.logging_level = logging_level
        self.prediction_dir = prediction_dir
        self.checkpoint_dir = checkpoint_dir
        if self.enable_parameter_averaging:
            self.checkpoint_dir_averaged = checkpoint_dir + '_avg'

        self.init_logging(self.log_dir)
        logging.info('\nnew run with parameters:\n{}'.format(pp.pformat(self.__dict__)))

        self.graph = self.build_graph()
        self.session = tf.Session(graph=self.graph)
        logging.info('built graph')

    def update_train_params(self):
        self.batch_size = self.batch_sizes[self.restart_idx]
        self.learning_rate = self.learning_rates[self.restart_idx]
        self.beta1_decay = self.beta1_decays[self.restart_idx]
        self.early_stopping_steps = self.patiences[self.restart_idx]

    def calculate_loss(self):
        raise NotImplementedError('subclass must implement this')

    def fit(self):
        with self.session.as_default():

            if self.warm_start_init_step:
                self.restore(self.warm_start_init_step)
                step = self.warm_start_init_step
            else:
                self.session.run(self.init)
                step = 0

            train_generator = self.reader.train_batch_generator(self.batch_size)
            val_generator = self.reader.val_batch_generator(self.validation_batch_size)

            train_loss_history = deque(maxlen=self.loss_averaging_window)
            val_loss_history = deque(maxlen=self.loss_averaging_window)
            train_time_history = deque(maxlen=self.loss_averaging_window)
            val_time_history = deque(maxlen=self.loss_averaging_window)
            if not hasattr(self, 'metrics'):
                self.metrics = {}

            metric_histories = {
                metric_name: deque(maxlen=self.loss_averaging_window) for metric_name in self.metrics
            }
            best_validation_loss, best_validation_tstep = float('inf'), 0

            while step < self.num_training_steps:

                # validation evaluation
                val_start = time.time()
                val_batch_df = next(val_generator)
                val_feed_dict = {
                    getattr(self, placeholder_name, None): data
                    for placeholder_name, data in val_batch_df.items() if hasattr(self, placeholder_name)
                }

                val_feed_dict.update({self.learning_rate_var: self.learning_rate, self.beta1_decay_var: self.beta1_decay})
                if hasattr(self, 'keep_prob'):
                    val_feed_dict.update({self.keep_prob: 1.0})
                if hasattr(self, 'is_training'):
                    val_feed_dict.update({self.is_training: False})

                results = self.session.run(
                    fetches=[self.loss] + self.metrics.values(),
                    feed_dict=val_feed_dict
                )
                val_loss = results[0]
                val_metrics = results[1:] if len(results) > 1 else []
                val_metrics = dict(zip(self.metrics.keys(), val_metrics))
                val_loss_history.append(val_loss)
                val_time_history.append(time.time() - val_start)
                for key in val_metrics:
                    metric_histories[key].append(val_metrics[key])

                if hasattr(self, 'monitor_tensors'):
                    for name, tensor in self.monitor_tensors.items():
                        [np_val] = self.session.run([tensor], feed_dict=val_feed_dict)
                        print(name)
                        print('min', np_val.min())
                        print('max', np_val.max())
                        print('mean', np_val.mean())
                        print('std', np_val.std())
                        print('nans', np.isnan(np_val).sum())
                        print()
                    print()
                    print()

                # train step
                train_start = time.time()
                train_batch_df = next(train_generator)
                train_feed_dict = {
                    getattr(self, placeholder_name, None): data
                    for placeholder_name, data in train_batch_df.items() if hasattr(self, placeholder_name)
                }

                train_feed_dict.update({self.learning_rate_var: self.learning_rate, self.beta1_decay_var: self.beta1_decay})
                if hasattr(self, 'keep_prob'):
                    train_feed_dict.update({self.keep_prob: self.keep_prob_scalar})
                if hasattr(self, 'is_training'):
                    train_feed_dict.update({self.is_training: True})

                train_loss, _ = self.session.run(
                    fetches=[self.loss, self.step],
                    feed_dict=train_feed_dict
                )
                train_loss_history.append(train_loss)
                train_time_history.append(time.time() - train_start)

                if step % self.log_interval == 0:
                    avg_train_loss = sum(train_loss_history) / len(train_loss_history)
                    avg_val_loss = sum(val_loss_history) / len(val_loss_history)
                    avg_train_time = sum(train_time_history) / len(train_time_history)
                    avg_val_time = sum(val_time_history) / len(val_time_history)
                    metric_log = (
                        "[[step {:>8}]]     "
                        "[[train {:>4}s]]     loss: {:<12}     "
                        "[[val {:>4}s]]     loss: {:<12}     "
                    ).format(
                        step,
                        round(avg_train_time, 4),
                        round(avg_train_loss, 8),
                        round(avg_val_time, 4),
                        round(avg_val_loss, 8),
                    )
                    early_stopping_metric = avg_val_loss
                    for metric_name, metric_history in metric_histories.items():
                        metric_val = sum(metric_history) / len(metric_history)
                        metric_log += '{}: {:<4}     '.format(metric_name, round(metric_val, 4))
                        if metric_name == self.early_stopping_metric:
                            early_stopping_metric = metric_val

                    logging.info(metric_log)

                    if early_stopping_metric < best_validation_loss:
                        best_validation_loss = early_stopping_metric
                        best_validation_tstep = step
                        if step > self.min_steps_to_checkpoint:
                            self.save(step)
                            if self.enable_parameter_averaging:
                                self.save(step, averaged=True)

                    if step - best_validation_tstep > self.early_stopping_steps:

                        if self.num_restarts is None or self.restart_idx >= self.num_restarts:
                            logging.info('best validation loss of {} at training step {}'.format(
                                best_validation_loss, best_validation_tstep))
                            logging.info('early stopping - ending training.')
                            return

                        if self.restart_idx < self.num_restarts:
                            self.restore(best_validation_tstep)
                            step = best_validation_tstep
                            self.restart_idx += 1
                            self.update_train_params()
                            train_generator = self.reader.train_batch_generator(self.batch_size)

                step += 1

            if step <= self.min_steps_to_checkpoint:
                best_validation_tstep = step
                self.save(step)
                if self.enable_parameter_averaging:
                    self.save(step, averaged=True)

            logging.info('num_training_steps reached - ending training')

    def predict(self, chunk_size=256):
        if not os.path.isdir(self.prediction_dir):
            os.makedirs(self.prediction_dir)

        if hasattr(self, 'prediction_tensors'):
            prediction_dict = {tensor_name: [] for tensor_name in self.prediction_tensors}

            test_generator = self.reader.test_batch_generator(chunk_size)
            for i, test_batch_df in enumerate(test_generator):
                if i % 10 == 0:
                    print(i*len(test_batch_df))

                test_feed_dict = {
                    getattr(self, placeholder_name, None): data
                    for placeholder_name, data in test_batch_df.items() if hasattr(self, placeholder_name)
                }
                if hasattr(self, 'keep_prob'):
                    test_feed_dict.update({self.keep_prob: 1.0})
                if hasattr(self, 'is_training'):
                    test_feed_dict.update({self.is_training: False})

                tensor_names, tf_tensors = zip(*self.prediction_tensors.items())
                np_tensors = self.session.run(
                    fetches=tf_tensors,
                    feed_dict=test_feed_dict
                )
                for tensor_name, tensor in zip(tensor_names, np_tensors):
                    prediction_dict[tensor_name].append(tensor)

            for tensor_name, tensor in prediction_dict.items():
                np_tensor = np.concatenate(tensor, 0)
                save_file = os.path.join(self.prediction_dir, '{}.npy'.format(tensor_name))
                logging.info('saving {} with shape {} to {}'.format(tensor_name, np_tensor.shape, save_file))
                np.save(save_file, np_tensor)

        if hasattr(self, 'parameter_tensors'):
            for tensor_name, tensor in self.parameter_tensors.items():
                np_tensor = tensor.eval(self.session)

                save_file = os.path.join(self.prediction_dir, '{}.npy'.format(tensor_name))
                logging.info('saving {} with shape {} to {}'.format(tensor_name, np_tensor.shape, save_file))
                np.save(save_file, np_tensor)

    def save(self, step, averaged=False):
        saver = self.saver_averaged if averaged else self.saver
        checkpoint_dir = self.checkpoint_dir_averaged if averaged else self.checkpoint_dir
        if not os.path.isdir(checkpoint_dir):
            logging.info('creating checkpoint directory {}'.format(checkpoint_dir))
            os.mkdir(checkpoint_dir)

        model_path = os.path.join(checkpoint_dir, 'model')
        logging.info('saving model to {}'.format(model_path))
        saver.save(self.session, model_path, global_step=step)

    def restore(self, step=None, averaged=False):
        saver = self.saver_averaged if averaged else self.saver
        checkpoint_dir = self.checkpoint_dir_averaged if averaged else self.checkpoint_dir
        if not step:
            model_path = tf.train.latest_checkpoint(checkpoint_dir)
            logging.info('restoring model parameters from {}'.format(model_path))
            saver.restore(self.session, model_path)
        else:
            model_path = os.path.join(
                checkpoint_dir, 'model{}-{}'.format('_avg' if averaged else '', step)
            )
            logging.info('restoring model from {}'.format(model_path))
            saver.restore(self.session, model_path)

    def init_logging(self, log_dir):
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
        log_file = 'log_{}.txt'.format(date_str)

        try:                 # Python 2
            reload(logging)  # bad
        except NameError:    # Python 3
            import logging
        logging.basicConfig(
            filename=os.path.join(log_dir, log_file),
            level=self.logging_level,
            format='[[%(asctime)s]] %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        logging.getLogger().addHandler(logging.StreamHandler())

    def update_parameters(self, loss):
        if self.regularization_constant != 0:
            l2_norm = tf.reduce_sum([tf.sqrt(tf.reduce_sum(tf.square(param))) for param in tf.trainable_variables()])
            loss = loss + self.regularization_constant*l2_norm

        optimizer = self.get_optimizer(self.learning_rate_var, self.beta1_decay_var)
        grads = optimizer.compute_gradients(loss)
        clipped = [(tf.clip_by_value(g, -self.grad_clip, self.grad_clip), v_) for g, v_ in grads]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            step = optimizer.apply_gradients(clipped, global_step=self.global_step)

        if self.enable_parameter_averaging:
            maintain_averages_op = self.ema.apply(tf.trainable_variables())
            with tf.control_dependencies([step]):
                self.step = tf.group(maintain_averages_op)
        else:
            self.step = step

        logging.info('all parameters:')
        logging.info(pp.pformat([(var.name, shape(var)) for var in tf.global_variables()]))

        logging.info('trainable parameters:')
        logging.info(pp.pformat([(var.name, shape(var)) for var in tf.trainable_variables()]))

        logging.info('trainable parameter count:')
        logging.info(str(np.sum(np.prod(shape(var)) for var in tf.trainable_variables())))

    def get_optimizer(self, learning_rate, beta1_decay):
        if self.optimizer == 'adam':
            return tf.train.AdamOptimizer(learning_rate, beta1=beta1_decay)
        elif self.optimizer == 'gd':
            return tf.train.GradientDescentOptimizer(learning_rate)
        elif self.optimizer == 'rms':
            return tf.train.RMSPropOptimizer(learning_rate, decay=beta1_decay, momentum=0.9)
        else:
            assert False, 'optimizer must be adam, gd, or rms'

    def build_graph(self):
        with tf.Graph().as_default() as graph:
            self.ema = tf.train.ExponentialMovingAverage(decay=0.99)
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate_var = tf.Variable(0.0, trainable=False)
            self.beta1_decay_var = tf.Variable(0.0, trainable=False)

            self.loss = self.calculate_loss()
            self.update_parameters(self.loss)

            self.saver = tf.train.Saver(max_to_keep=1)
            if self.enable_parameter_averaging:
                self.saver_averaged = tf.train.Saver(self.ema.variables_to_restore(), max_to_keep=1)

            self.init = tf.global_variables_initializer()
            return graph

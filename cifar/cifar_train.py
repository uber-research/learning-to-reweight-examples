# Copyright (c) 2017 - 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
# Trains a CNN on CIFAR with noisy labels (uniform flip).
#
# Usage:
# python cifar.cifar_train
#
# Mandatory flags:
#   --config           [string]        Config file path
#
# Optional Flags:
#   --baseline                         Run the baseline model directly (regular training)
#   --eval                             Run evaluation only
#   --finetune                         Run finetuning on the pretrained model (for 5k steps or 2k steps on clean only)
#   --random_weight                    Run the random weight baseline
#   --restore                          Whether to restore model
#   --verbose                          Whether to print extra information
#   --noise_ratio      [float]         Percentage of noise, default 0.4
#   --bsize_a          [int]           Batch size of the noisy pass, default 100
#   --bsize_b          [int]           Batch size of the clean pass, default 100
#   --eval_interval    [int]           Number of steps per evaluation, default 1000
#   --log_interval     [int]           Number of steps to print loss values, default 10
#   --num_clean        [int]           Total number of clean images in the training set, default 100
#   --num_val          [int]           Total number of images in the validation set, default 5000
#   --save_interval    [int]           Number of steps to save a checkpoint, default 10000
#   --seed             [int]           Random seed, default 0
#   --id               [string]        Experiment ID, for restoring purpose
#   --data_root        [string]        Data folder path, default "./data"
#   --results          [string]        Model checkpoint saving path, default "./results"
#
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import os
import six
import tensorflow as tf

from collections import namedtuple
from google.protobuf.text_format import Merge, MessageToString
from tqdm import tqdm

from datasets.data_factory import get_data_inputs
from models.model_factory import get_model
from models.cnn.configs.resnet_model_config_pb2 import ResnetModelConfig
from models.cnn.resnet_model import ResnetModel    # NOQA
from utils import logger, gen_id
from utils.learn_rate_schedulers import FixedLearnRateScheduler

from cifar.generate_noisy_cifar_data import generate_noisy_cifar
from cifar.noisy_cifar_dataset import NoisyCifar100Dataset    # NOQA
from cifar.noisy_cifar_dataset import NoisyCifar10Dataset    # NOQA
from cifar.noisy_cifar_input_pipeline import NoisyCifarInputPipeline    # NOQA
from models.assigned_weights_resnet_model import AssignedWeightsResnetModel    # NOQA
from models.weighted_resnet_model import WeightedResnetModel    # NOQA
from models.reweight_model import reweight_autodiff

log = logger.get()


def _get_config():
    """Gets config."""
    config_file = FLAGS.config
    config = ResnetModelConfig()
    assert config_file is not None, 'Must pass in a configuration file through --config'
    Merge(open(config_file).read(), config)
    return config


def _get_model(config, inp, label, bsize, is_training, name_scope, reuse):
    """Builds models."""
    model_cls = 'resnet'
    trn_kwargs = {
        'is_training': is_training,
        'inp': inp,
        'label': label,
        'batch_size': bsize,
    }
    with tf.name_scope(name_scope):
        with tf.variable_scope('Model', reuse=reuse):
            m = get_model(model_cls, config, **trn_kwargs)
    return m


def _get_assign_weighted_model(config, inp, label, weights, weights_dict, bsize, is_training,
                               name_scope, reuse):
    """Builds models."""
    model_cls = 'assign-wts-resnet'
    trn_kwargs = {
        'is_training': is_training,
        'inp': inp,
        'label': label,
        'ex_wts': weights,
        'batch_size': bsize
    }
    with tf.name_scope(name_scope):
        with tf.variable_scope('Model', reuse=reuse):
            m = get_model(model_cls, config, weights_dict, **trn_kwargs)
    return m


def _get_reweighted_model(config, inp, label, weights, bsize, is_training, name_scope, reuse):
    """Builds models."""
    model_cls = 'weighted-resnet'
    trn_kwargs = {
        'is_training': is_training,
        'inp': inp,
        'label': label,
        'ex_wts': weights,
        'batch_size': bsize
    }
    with tf.name_scope(name_scope):
        with tf.variable_scope('Model', reuse=reuse):
            m = get_model(model_cls, config, **trn_kwargs)
    return m


def _get_data_input(dataset, data_dir, split, bsize, is_training, seed, **kwargs):
    """Builds data input."""
    data = get_data_inputs(dataset, data_dir, split, is_training, bsize, 'cifar-noisy', **kwargs)
    batch = data.inputs(seed=seed)
    inp, label, idx, clean_flag = batch['image'], batch['label'], batch['index'], batch['clean']
    DataTuple = namedtuple('DataTuple', ['data', 'inputs', 'labels', 'index', 'clean_flag'])
    return DataTuple(data=data, inputs=inp, labels=label, index=idx, clean_flag=clean_flag)


def _get_data_inputs(bsize, seed=0):
    """Gets data input tensors."""
    # Compute the dataset directory for this experiment.
    data_name = FLAGS.dataset
    data_dir = os.path.join(FLAGS.data_root, data_name)
    print(data_dir)

    log.info('Building dataset')
    trn_data = _get_data_input(data_name, data_dir, 'train', bsize, True, seed)
    val_data = _get_data_input(data_name, data_dir, 'validation', bsize, False, seed)
    test_data = _get_data_input(data_name, data_dir, 'test', bsize, False, seed)

    class Datasets:
        train = trn_data
        val = val_data
        test = test_data

    return Datasets()


def _get_noisy_data_inputs(bsize, seed=0):
    """Gets data input tensors."""
    # Compute the dataset directory for this experiment.
    data_name = FLAGS.dataset + '-noisy-clean{:d}-noise{:d}-val{:d}-seed{:d}'.format(
        FLAGS.num_clean, int(FLAGS.noise_ratio * 100), FLAGS.num_val, FLAGS.seed)
    data_dir = os.path.join(FLAGS.data_root, data_name)
    print(data_dir)

    # Generate TF records if not exist.
    if not os.path.exists(data_dir):
        generate_noisy_cifar(FLAGS.dataset,
                             os.path.join(FLAGS.data_root, FLAGS.dataset), FLAGS.num_val,
                             FLAGS.noise_ratio, FLAGS.num_clean, data_dir, FLAGS.seed)

    log.info('Building dataset')
    dataset = FLAGS.dataset + '-noisy'
    trn_clean_data = _get_data_input(
        dataset,
        data_dir,
        'train_clean',
        bsize,
        True,
        seed,
        num_val=FLAGS.num_val,
        num_clean=FLAGS.num_clean)
    trn_noisy_data = _get_data_input(
        dataset,
        data_dir,
        'train_noisy',
        bsize,
        True,
        seed,
        num_val=FLAGS.num_val,
        num_clean=FLAGS.num_clean)
    val_data = _get_data_input(
        dataset,
        data_dir,
        'validation',
        bsize,
        False,
        seed,
        num_val=FLAGS.num_val,
        num_clean=FLAGS.num_clean)
    val_noisy_data = _get_data_input(
        dataset,
        data_dir,
        'validation_noisy',
        bsize,
        False,
        seed,
        num_val=FLAGS.num_val,
        num_clean=FLAGS.num_clean)
    test_data = _get_data_input(
        dataset,
        data_dir,
        'test',
        bsize,
        False,
        seed,
        num_val=FLAGS.num_val,
        num_clean=FLAGS.num_clean)

    class Datasets:
        train_clean = trn_clean_data
        train_noisy = trn_noisy_data
        val = val_data
        val_noisy = val_noisy_data
        test = test_data

    return Datasets()


def _get_exp_logger(sess, log_folder):
    """Gets a TensorBoard logger."""
    with tf.name_scope('Summary'):
        writer = tf.summary.FileWriter(os.path.join(log_folder, 'logs'), sess.graph)
        summaries = dict()

    class ExperimentLogger():
        def log(self, niter, name, value):
            summary = tf.Summary()
            summary.value.add(tag=name, simple_value=value)
            writer.add_summary(summary, niter)

        def flush(self):
            """Flushes results to disk."""

        def close(self):
            """Closes writer."""
            writer.close()

    return ExperimentLogger()


def train_step(sess, model_a, model_b, ex_weights, model_c, data_a, data_b, ex_wts_a=None):
    """Train step.

    :param sess:
    :param model_a:
    :param model_b:
    :param ex_weights:
    :param model_c:
    :param data_a:
    :param data_b:
    :param ex_wts_a:

    :return: Training cross entropy.
    """
    # Get data.
    if not FLAGS.baseline:
        inp_a, label_a, idx_a = sess.run([data_a.inputs, data_a.labels, data_a.index])
        dict_a = model_a._get_feed_dict(inp=inp_a, label=label_a)
        if FLAGS.random_weight:
            rnd_normal = np.random.normal(0.0, 1.0, [inp_a.shape[0]])
            rnd_normal_plus = np.maximum(rnd_normal, 0.0)
            ex_weights_value = rnd_normal_plus / rnd_normal_plus.sum()
        else:
            inp_b, label_b = sess.run([data_b.inputs, data_b.labels])
            dict_b = model_b._get_feed_dict(inp=inp_b, label=label_b)

            fdict = {}
            for kk in dict_a.keys():
                fdict[kk] = dict_a[kk]
            for kk in dict_b.keys():
                fdict[kk] = dict_b[kk]

            # Get example weights.
            ex_weights_value = sess.run(ex_weights, feed_dict=fdict)
    else:
        # Baseline mode.
        inp_a, label_a = sess.run([data_a.inputs, data_a.labels])
        ex_weights_value = np.ones([inp_a.shape[0]]) / float(inp_a.shape[0])

    # Run weighted loss.
    return model_c.train_step(sess, inp=inp_a, label=label_a, ex_wts=ex_weights_value)


def finetune_step(sess, model, data):
    inp, label = sess.run([data.inputs, data.labels])
    ex_weights_value = np.ones([inp.shape[0]]) / float(inp.shape[0])
    return model.train_step(sess, inp=inp, label=label, ex_wts=ex_weights_value)


def evaluate(sess, model, num_batches, data=None):
    """Runs evaluation."""
    num_correct = 0.0
    count = 0.0
    avg_ce = 0.0
    for ii in six.moves.xrange(num_batches):
        if data is not None:
            inp, label = sess.run([data.inputs, data.labels])
        else:
            inp, label = None, None
        correct, ce = model.eval_step(sess, inp=inp, label=label)
        num_correct += correct.sum()
        count += correct.size
        avg_ce += ce
    acc = (num_correct / count)
    avg_ce /= float(num_batches)
    return acc, avg_ce


def evaluate_train(sess, model, num_batches, data=None):
    """evaluate training model on clean and noisy data"""
    num_correct_clean = 0.0
    num_clean = 0.0
    num_correct_noise = 0.0
    num_noise = 0.0
    for ii in six.moves.xrange(num_batches):
        if data is not None:
            inp, label, clean_flag = sess.run([data.inputs, data.labels, data.clean_flag])
        else:
            inp, label = None, None
        correct, _ = model.eval_step(sess, inp=inp, label=label)
        result_clean = correct * clean_flag
        result_noise = correct * (1 - clean_flag)
        num_correct_clean += result_clean.sum()
        num_correct_noise += result_noise.sum()
        num_clean += clean_flag.sum()
        num_noise += (1 - clean_flag).sum()
    acc_clean = (num_correct_clean / num_clean)
    acc_noise = (num_correct_noise / num_noise)
    return acc_clean, num_clean, acc_noise, num_noise


def save(sess, saver, global_step, config, save_folder):
    """Snapshots a model."""
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    config_file = os.path.join(save_folder, 'config.prototxt')
    with open(config_file, 'w') as f:
        f.write(MessageToString(config))
    log.info('Saving to {}'.format(save_folder))
    saver.save(sess, os.path.join(save_folder, 'model.ckpt'), global_step=global_step)


def finetune_model(sess, exp_id, config, trn_data, data, model, val_model, save_folder=None):
    mvalid = val_model

    # Train loop.
    ce = 0.0
    trn_acc = 0.0
    val_acc = 0.0
    saver = tf.train.Saver()
    # Logger.
    if save_folder is not None:
        exp_logger = _get_exp_logger(sess, save_folder)

    niter_start = int(model.global_step.eval())
    w_list = tf.trainable_variables()
    log.info('Model initialized.')
    num_params = np.array([np.prod(np.array([int(ss) for ss in w.get_shape()]))
                           for w in w_list]).sum()
    log.info('Number of parameters {}'.format(num_params))

    # Set up learning rate schedule.
    if not FLAGS.restore:
        lr_list = [1000]    # For clean only model.
    else:
        lr_list = config.optimizer_config.learn_rate_list
    if config.optimizer_config.learn_rate_scheduler_type == 'fixed':
        lr = config.optimizer_config.learn_rate
        lr_decay_steps = config.optimizer_config.learn_rate_decay_steps
        lr_scheduler = FixedLearnRateScheduler(sess, model, lr, lr_decay_steps, lr_list)
    else:
        raise Exception('Unknown learning rate scheduler {}'.format(
            config.optimizer_config.learn_rate_scheduler))

    if not FLAGS.restore:
        max_train_iter = 2000    # For clean only model.
    else:
        max_train_iter = niter_start + 5000
    it = tqdm(six.moves.xrange(niter_start, max_train_iter), desc=exp_id, ncols=0)
    for niter in it:
        lr_scheduler.step(niter)
        ce = finetune_step(sess, model, data)

        # Evaluate accuracy.
        if (niter + 1) % FLAGS.eval_interval == 0 or niter == niter_start:
            trn_acc, _ = evaluate(sess, model, 100, trn_data)
            val_acc, val_ce = evaluate(sess, val_model, 50)
            if save_folder is not None:
                exp_logger.log(niter + 1, 'train ft acc', trn_acc * 100.0)
                exp_logger.log(niter + 1, 'val ft acc', val_acc * 100.0)
                exp_logger.log(niter + 1, 'lr', lr_scheduler.learn_rate)
                if niter > 0:
                    exp_logger.log(niter + 1, 'val ft ce', val_ce)
                exp_logger.flush()

        # Save model.
        if (niter + 1) % FLAGS.save_interval == 0 or niter == 0:
            if save_folder is not None:
                save(sess, saver, model.global_step, config, save_folder)

        # Show on progress bar.
        if (niter + 1) % FLAGS.log_interval == 0 or niter == 0:
            if save_folder is not None:
                exp_logger.log(niter + 1, 'train ft ce', ce)
            disp_dict = {
                'ce': '{:.3e}'.format(ce),
                'train_acc': '{:.3f}%'.format(trn_acc * 100),
                'val_acc': '{:.3f}%'.format(val_acc * 100),
                'lr': '{:.3e}'.format(lr_scheduler.learn_rate)
            }
            it.set_postfix(**disp_dict)

    val_acc, _ = evaluate(sess, val_model, 50)

    if save_folder is not None:
        exp_logger.close()

    return val_acc


def train_model(sess,
                exp_id,
                config,
                data_a,
                data_b,
                model_a,
                model_b,
                ex_weights,
                model_c,
                val_model,
                ex_wts_a=None,
                noisy_val_model=None,
                save_folder=None):
    """
    Trains a CIFAR model.

    :param sess:                 [Session]    Session object.
    :param exp_id:               [string]     Experiment ID.
    :param config:               [object]     Config object.
    :param data_a:               [object]     Training set A.
    :param data_b:               [object]     Training set B.
    :param model_a:              [object]     Model A.
    :param model_b:              [object]     Model B.
    :param ex_weights:           [Tensor]     Example weights.
    :param trn_reweighted_model: [object]     Reweighted training model.
    :param val_model:            [object]     Validation model.
    :param val_noisy_model:      [object]     Noisy validation model.
    :param save_folder:          [string]     Folder to save model.

    :return:                     [float]      Final validation accuracy.
    """
    mvalid = val_model

    # Train loop.
    ce = 0.0
    trn_acc = 0.0
    val_acc = 0.0
    saver = tf.train.Saver()
    # Logger.
    if save_folder is not None:
        exp_logger = _get_exp_logger(sess, save_folder)

    niter_start = int(model_c.global_step.eval())
    w_list = tf.trainable_variables()
    log.info('Model initialized.')
    num_params = np.array([np.prod(np.array([int(ss) for ss in w.get_shape()]))
                           for w in w_list]).sum()
    log.info('Number of parameters {}'.format(num_params))

    # Set up learning rate schedule.
    if config.optimizer_config.learn_rate_scheduler_type == 'fixed':
        lr = config.optimizer_config.learn_rate
        lr_decay_steps = config.optimizer_config.learn_rate_decay_steps
        lr_list = config.optimizer_config.learn_rate_list
        lr_list = [x for x in lr_list]
        lr_scheduler = FixedLearnRateScheduler(sess, model_c, lr, lr_decay_steps, lr_list)
    else:
        raise Exception('Unknown learning rate scheduler {}'.format(
            config.optimizer_config.learn_rate_scheduler))

    max_train_iter = config.optimizer_config.max_train_iter
    it = tqdm(six.moves.xrange(niter_start, max_train_iter), desc=exp_id, ncols=0)
    for niter in it:
        lr_scheduler.step(niter)
        ce = train_step(
            sess, model_a, model_b, ex_weights, model_c, data_a, data_b, ex_wts_a=ex_wts_a)

        # Evaluate accuracy.
        if (niter + 1) % FLAGS.eval_interval == 0 or niter == 0:
            trn_acc, _ = evaluate(sess, model_c, 100, data_a)
            val_acc, val_ce = evaluate(sess, val_model, 50)
            if noisy_val_model is not None:
                noisy_val_acc, noisy_val_ce = evaluate(sess, noisy_val_model, 50)
            if save_folder is not None:
                exp_logger.log(niter + 1, 'train acc', trn_acc * 100.0)
                exp_logger.log(niter + 1, 'val acc', val_acc * 100.0)
                if noisy_val_model is not None:
                    exp_logger.log(niter + 1, 'val noisy acc', noisy_val_acc * 100.0)
                exp_logger.log(niter + 1, 'lr', lr_scheduler.learn_rate)
                if niter > 0:
                    exp_logger.log(niter + 1, 'val ce', val_ce)
                exp_logger.flush()

        # Save model.
        if (niter + 1) % FLAGS.save_interval == 0 or niter == 0:
            if save_folder is not None:
                save(sess, saver, model_c.global_step, config, save_folder)

        # Show on progress bar.
        if (niter + 1) % FLAGS.log_interval == 0 or niter == 0:
            if save_folder is not None:
                exp_logger.log(niter + 1, 'train ce', ce)
            disp_dict = {
                'ce': '{:.3e}'.format(ce),
                'train_acc': '{:.3f}%'.format(trn_acc * 100),
                'val_acc': '{:.3f}%'.format(val_acc * 100),
                'lr': '{:.3e}'.format(lr_scheduler.learn_rate)
            }
            if noisy_val_model is not None:
                disp_dict['val_noise_acc'] = '{:.3f}%'.format(noisy_val_acc * 100)
            it.set_postfix(**disp_dict)

    val_acc, _ = evaluate(sess, val_model, 50)
    if noisy_val_model is not None:
        noisy_val_acc, _ = evaluate(sess, noisy_val_model, 50)
    else:
        noisy_val_acc = 0.0

    if save_folder is not None:
        exp_logger.close()

    return val_acc, noisy_val_acc


def main():
    # -----------------------------------------------------------------
    # Loads parammeters.
    config = _get_config()

    # bsize = config.optimizer_config.batch_size
    bsize_a = FLAGS.bsize_a
    bsize_b = FLAGS.bsize_b

    log.info('Config: {}'.format(MessageToString(config)))

    data_name = FLAGS.dataset + '-noisy-clean{:d}-noise{:d}-val{:d}-seed{:d}'.format(
        FLAGS.num_clean, int(FLAGS.noise_ratio * 100), FLAGS.num_val, FLAGS.seed)
    data_dir = os.path.join(FLAGS.data_root, data_name)

    # Initializes variables.
    np.random.seed(0)
    if not hasattr(config, 'seed'):
        tf.set_random_seed(1234)
        log.info('Setting tensorflow random seed={:d}'.format(1234))
    else:
        log.info('Setting tensorflow random seed={:d}'.format(config.seed))
        tf.set_random_seed(config.seed)

    with tf.Graph().as_default(), tf.Session() as sess:
        # -----------------------------------------------------------------
        # Building datasets.
        dataset_a = _get_noisy_data_inputs(bsize_a, seed=0)
        if bsize_a != bsize_b:
            dataset_b = _get_noisy_data_inputs(
                bsize_b, seed=1000)    # Make sure they have different seeds.
        else:
            dataset_b = dataset_a
        data_a = dataset_a.train_noisy
        data_b = dataset_b.train_clean
        # Assign number of classes to model config.
        config.resnet_module_config.num_classes = dataset_a.test.data.dataset.num_classes()
        log.info('Weight decay {:.3e}'.format(config.resnet_module_config.weight_decay))

        # -----------------------------------------------------------------
        # Experiment ID.
        if FLAGS.id is None:
            dataset_name = FLAGS.dataset
            if FLAGS.config is not None:
                model_name = FLAGS.config.split('/')[-1].split('.')[0][6:]
            else:
                model_name = FLAGS.model
            exp_id = 'e_' + dataset_name + '_' + model_name
            exp_id = gen_id.get(exp_id)
        else:
            exp_id = FLAGS.id
            dataset_name = exp_id.split('_')[1]

        # -----------------------------------------------------------------
        # Save folder.
        if FLAGS.results is not None:
            save_folder = os.path.realpath(os.path.abspath(os.path.join(FLAGS.results, exp_id)))
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
        else:
            save_folder = None

        # -----------------------------------------------------------------
        # Build shared-weights model replicas, for different data sources.
        model_c = _get_reweighted_model(config, None, None, None, None, True, 'TrainC', None)
        var_list = tf.trainable_variables()
        [print(vv.name) for vv in var_list]

        def build_model_a(wts_dict, ex_wts, reuse):
            return _get_assign_weighted_model(config, None, None, ex_wts, wts_dict, bsize_a, True,
                                              'TrainA', reuse)

        def build_model_b(wts_dict, ex_wts, reuse):
            return _get_assign_weighted_model(config, None, None, ex_wts, wts_dict, bsize_b, True,
                                              'TrainB', reuse)

        ex_wts_a = tf.placeholder_with_default(
            tf.zeros([bsize_a], dtype=tf.float32), [bsize_a], name='ex_wts_a')

        model_a, model_b, ex_weights = reweight_autodiff(
            build_model_a,
            build_model_b,
            bsize_a,
            bsize_b,
            ex_wts_a=ex_wts_a)
        val_model = _get_model(config, dataset_a.val.inputs, dataset_a.val.labels, bsize_a, False,
                               'Val', True)
        noisy_val_model = _get_model(config, dataset_a.val_noisy.inputs, dataset_a.val_noisy.labels,
                                     bsize_a, False, 'NoisyVal', True)
        test_model = _get_model(config, dataset_a.test.inputs, dataset_a.test.labels, bsize_a,
                                False, 'Test', True)

        saver = tf.train.Saver()
        if FLAGS.restore:
            log.info('Restore checkpoint \'{}\''.format(save_folder))
            saver.restore(sess, tf.train.latest_checkpoint(save_folder))
        else:
            sess.run(tf.global_variables_initializer())

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # -----------------------------------------------------------------
        # Train a model.
        if not FLAGS.eval and not FLAGS.finetune:
            val_acc = train_model(
                sess,
                exp_id,
                config,
                data_a,
                data_b,
                model_a,
                model_b,
                ex_weights,
                model_c,
                val_model,
                ex_wts_a=ex_wts_a,
                noisy_val_model=noisy_val_model,
                save_folder=save_folder)

        if FLAGS.finetune:
            val_acc = finetune_model(
                sess, exp_id, config, data_a, data_b, model_c, val_model, save_folder=save_folder)

        # -----------------------------------------------------------------
        # Final test.
        train_acc, _ = evaluate(sess, model_c, 450, data_a)
        val_acc, _ = evaluate(sess, val_model, 50)
        if noisy_val_model is not None:
            noisy_val_acc, _ = evaluate(sess, noisy_val_model, 50)
        test_acc, _ = evaluate(sess, test_model, 100)
        coord.request_stop()
        coord.join(threads)
    log.info('Final train accuracy = {:.3f}'.format(train_acc * 100))
    log.info('Final val accuracy = {:.3f}'.format(val_acc * 100))
    if noisy_val_model is not None:
        log.info('Final noisy val accuracy = {:.3f}'.format(noisy_val_acc * 100))
    log.info('Final test accuracy = {:.3f}'.format(test_acc * 100))


if __name__ == '__main__':
    flags = tf.flags
    flags.DEFINE_bool('baseline', False, 'Run non-reweighting baseline')
    flags.DEFINE_bool('eval', False, 'Whether run evaluation only')
    flags.DEFINE_bool('finetune', False, 'Whether to finetune model')
    flags.DEFINE_bool('random_weight', False, 'Use random weights')
    flags.DEFINE_bool('restore', False, 'Whether restore model')
    flags.DEFINE_bool('verbose', True, 'Whether to show logging.INFO')
    flags.DEFINE_float('noise_ratio', 0.4, 'Noise ratio in the noisy training set')
    flags.DEFINE_integer('bsize_a', 100, 'Batch size multiplier for data A')
    flags.DEFINE_integer('bsize_b', 100, 'Batch size multiplier for data B')
    flags.DEFINE_integer('eval_interval', 1000, 'Number of steps between evaluations')
    flags.DEFINE_integer('log_interval', 10, 'Interval for writing loss values to TensorBoard')
    flags.DEFINE_integer('num_clean', 100, 'Number of clean images in the training set')
    flags.DEFINE_integer('num_val', 5000, 'Number of validation images')
    flags.DEFINE_integer('save_interval', 10000, 'Number of steps between checkpoints')
    flags.DEFINE_integer('seed', 0, 'Random seed for creating the split')
    flags.DEFINE_string('config', None, 'Manually defined config file')
    flags.DEFINE_string('data_root', './data', 'Data folder')
    flags.DEFINE_string('dataset', 'cifar-10', 'Dataset name')
    flags.DEFINE_string('id', None, 'Experiment ID')
    flags.DEFINE_string('results', './results/cifar', 'Saving folder')
    FLAGS = flags.FLAGS
    if FLAGS.verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.WARNING)
    main()

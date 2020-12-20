import argparse
import gc
import logging
import os
import sys
import time

from collections import defaultdict
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# for debug
import cv2
import numpy as np
import matplotlib.pyplot as plt
# for debug

from intention_prediction.scripts.data.loaderJAAD import data_loader
from intention_prediction.scripts.losses import gan_g_loss, gan_d_loss, l2_loss

#from intention_prediction.scripts.models import CNNLSTM1 as CNNLSTM  # , CNNMP
from intention_prediction.scripts.models import CNNLSTMJAAD2 as CNNLSTM  # , CNNMP
from intention_prediction.scripts.utils import int_tuple, bool_flag, get_total_norm
from intention_prediction.scripts.utils import relative_to_abs, get_dset_path

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--dataset', default='./datasets/lausanne', type=str)
parser.add_argument('--loader_num_workers', default=1, type=int)
parser.add_argument('--timestep', default=30, type=int)
parser.add_argument('--obs_len', default=8, type=int)

# Optimization
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_iterations', default=10000, type=int)
parser.add_argument('--num_epochs', default=200, type=int)

# Model Options
parser.add_argument('--embedding_dim', default=64, type=int)
parser.add_argument('--h_dim', default=32, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag)
parser.add_argument('--mlp_dim', default=64, type=int)
parser.add_argument('--learning_rate', default=0.0001, type=float)

# Output
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--print_every', default=5, type=int)
parser.add_argument('--checkpoint_every', default=100, type=int)
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=1, type=int)

# Misc
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="0", type=str)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
        float_dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Floattensor
    return long_dtype, float_dtype


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    # Build the training set
    logger.info("Initializing train set")
    train_path = os.path.join(args.dataset, "train")
    train_dset, train_loader = data_loader(args, train_path, "train")

    # Build the validation set
    logger.info("Initializing val set")
    val_path = os.path.join(args.dataset, "val")
    val_dset, val_loader = data_loader(args, val_path, "val")

    # set data type to cpu/gpu
    long_dtype, float_dtype = get_dtypes(args)

    iterations_per_epoch = train_dset / args.batch_size
    if args.num_epochs:
        args.num_iterations = int(iterations_per_epoch * args.num_epochs)

    logger.info('There are {} iterations per epoch'.format(iterations_per_epoch))

    # initialize the CNN LSTM
    classifier = CNNLSTM(
        embedding_dim=args.embedding_dim,
        h_dim=args.h_dim,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout)
    classifier.apply(init_weights)
    classifier.type(float_dtype).train()

    # set the optimizer
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)

    # define the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Maybe restore from checkpoint
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(args.output_dir,
                                    '%s_with_model.pt' % args.checkpoint_name)

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        classifier.load_state_dict(checkpoint['classifier_state'])
        optimizer.load_state_dict(checkpoint['classifier_optim_state'])
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
        checkpoint['restore_ts'].append(t)
    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'classifier_losses': defaultdict(list),  # classifier loss
            'losses_ts': [],  # loss at timestep ?
            'metrics_val': defaultdict(list),  # valid metrics (loss and accuracy)
            'metrics_train': defaultdict(list),  # train metrics (loss and accuracy)
            'sample_ts': [],
            'restore_ts': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'classifier_state': None,
            'classifier_optim_state': None,
            'classifier_best_state': None,
            'best_t': None,
        }
    t0 = None
    print("Total no of iterations: ", args.num_iterations)
    while t < args.num_iterations:

        gc.collect()
        epoch += 1
        logger.info('Starting epoch {}'.format(epoch))

        for batch in train_loader:

            # Maybe save a checkpoint
            if t == 0:
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)

                # Check stats on the validation set
                logger.info('Checking stats on val ...')
                metrics_val = check_accuracy(args, val_loader, classifier, loss_fn)
                logger.info('Checking stats on train ...')
                metrics_train = check_accuracy(args, train_loader, classifier, loss_fn)

                for k, v in sorted(metrics_val.items()):
                    logger.info('  [val] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_val'][k].append(v)
                for k, v in sorted(metrics_train.items()):
                    logger.info('  [train] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_train'][k].append(v)

                min_loss = min(checkpoint['metrics_val']['d_loss'])
                max_acc = max(checkpoint['metrics_val']['d_accuracy'])

                if metrics_val['d_loss'] == min_loss:
                    logger.info('New low for data loss')
                    checkpoint['best_t'] = t
                    checkpoint['best_state'] = classifier.state_dict()

                if metrics_val['d_accuracy'] == max_acc:
                    logger.info('New high for accuracy')
                    checkpoint['best_t'] = t
                    checkpoint['best_state'] = classifier.state_dict()

                # Save another checkpoint with model weights and optimizer state
                checkpoint['classifier_state'] = classifier.state_dict()
                checkpoint['classifier_optim_state'] = optimizer.state_dict()
                checkpoint_path = os.path.join(args.output_dir, '%s_with_model.pt' % args.checkpoint_name)
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                logger.info('Done.')

            # run batch and get losses
            losses = step(args, batch, classifier, loss_fn, optimizer)


            # measure time between batches
            if args.timing == 1:
                if t0 is not None:
                    logger.info('Interation {} took {}'.format(
                        t - 1, time.time() - t0
                    ))
                t0 = time.time()

            # Maybe save a checkpoint
            if t > 0 and t % args.checkpoint_every == 0:
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)

                # Check stats on the validation set
                logger.info('Checking stats on val ...')
                metrics_val = check_accuracy(args, val_loader, classifier, loss_fn)
                logger.info('Checking stats on train ...')
                metrics_train = check_accuracy(args, train_loader, classifier, loss_fn)

                for k, v in sorted(metrics_val.items()):
                    logger.info('  [val] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_val'][k].append(v)
                for k, v in sorted(metrics_train.items()):
                    logger.info('  [train] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_train'][k].append(v)

                min_loss = min(checkpoint['metrics_val']['d_loss'])
                max_acc = max(checkpoint['metrics_val']['d_accuracy'])

                if metrics_val['d_loss'] == min_loss:
                    logger.info('New low for data loss')
                    checkpoint['best_t'] = t
                    checkpoint['best_state'] = classifier.state_dict()

                if metrics_val['d_accuracy'] == max_acc:
                    logger.info('New high for accuracy')
                    checkpoint['best_t'] = t
                    checkpoint['best_state'] = classifier.state_dict()

                # Save another checkpoint with model weights and
                # optimizer state
                checkpoint['classifier_state'] = classifier.state_dict()
                checkpoint['classifier_optim_state'] = optimizer.state_dict()
                checkpoint_path = os.path.join(args.output_dir, '%s_with_model.pt' % args.checkpoint_name)
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                logger.info('Done.')

            t += 1
            if t >= args.num_iterations:
                # print best
                # print("[train] best accuracy ", checkpoint[]
                print("[train] best accuracy at lowest loss ",
                      checkpoint['metrics_train']['d_accuracy'][np.argmin(checkpoint['metrics_train']['d_loss'])])
                print("[train] best accuracy at highest accuracy ", max(checkpoint['metrics_train']['d_accuracy']))
                print("[val] best accuracy at lowest loss ",
                      checkpoint['metrics_val']['d_accuracy'][np.argmin(checkpoint['metrics_val']['d_loss'])])
                print("[val] best accuracy at highest accuracy ", max(checkpoint['metrics_val']['d_accuracy']))

                break


def step(args, batch, classifier, loss_fn, optimizer):
    (pedestrian_crops, standing_true, looking_true, walking_true, crossing_true, *_) = batch
    groundtruth = standing_true, looking_true, walking_true, crossing_true

    losses = {}
    loss = torch.zeros(1).type(torch.cuda.FloatTensor) if torch.cuda.is_available() else torch.zeros(1).type(torch.FloatTensor)

    # predict pedestrian decision
    predictions = classifier(pedestrian_crops)

    # compute loss
    data_loss_all = []
    for gt, pred in zip(groundtruth, predictions):
        data_loss_all.append(loss_fn(pred, standing_true.cuda()) if torch.cuda.is_available() else loss_fn(pred, gt.cpu()))

    loss_n = [data_loss.item() for data_loss in data_loss_all]
    data_loss_sum = torch.zeros(1).type(torch.cuda.FloatTensor) if torch.cuda.is_available() else torch.zeros(1).type(torch.FloatTensor)
    for data_loss in data_loss_all:
        data_loss_sum += data_loss
    avg = data_loss_sum / len(loss_n)
    losses['data_loss'] = avg
    loss += avg
    losses['total_loss'] = loss.item()

    # backprop given the loss
    optimizer.zero_grad()
    loss.backward()
    # if args.clipping_threshold > 0:
    #    nn.utils.clip_grad_norm_(classifier.parameters(),args.clipping_threshold)
    optimizer.step()

    return losses


def guided_backprop(args, loader, classifier,):
    data_confusions = []
    metrics = {}
    # classifier.eval()

    for batch in loader:
        # get batch
        (pedestrian_crops, decision_true, _) = batch

        print("batch size ", len(pedestrian_crops))
        print("timesteps", len(pedestrian_crops[0]))

        # predict decision
        decision_pred = classifier(pedestrian_crops, input_as_var=True)
        onehot_pred = torch.round(decision_pred.cpu())

        # print(classifier.gradients)

        # backprop
        classifier.zero_grad()
        decision_pred.backward(gradient=onehot_pred.cuda()) if torch.cuda.is_available() else decision_pred.backward(
            gradient=decision_pred.cpu())

    # classifier.train()
    return metrics


def check_accuracy(args, loader, classifier, loss_fn):
    losses = [[], [], [], []]

    confusions = [[], [], [], []]  #todo generalize
    classifier.eval()
    with torch.no_grad():
        for batch in loader:
            # get batch
            (pedestrian_crops, standing_true, looking_true, walking_true, crossing_true, *_) = batch

            # predict decision
            (standing_pred, looking_pred, walking_pred, crossing_pred) = classifier(pedestrian_crops)

            # compute loss
            standing_loss = loss_fn(standing_pred, standing_true)  #todo decision_true.cpu()
            looking_loss = loss_fn(looking_pred, looking_true)
            walking_loss = loss_fn(walking_pred, walking_true)
            crossing_loss = loss_fn(crossing_pred, crossing_true)

            losses[0].append(standing_loss.item())
            losses[1].append(looking_loss.item())
            losses[2].append(walking_loss.item())
            losses[3].append(crossing_loss.item())


            # build confusion matrix
            standing_confusion = confusion_matrix(standing_true.numpy(), standing_pred.max(1)[1].numpy())
            looking_confusion = confusion_matrix(looking_true.numpy(), looking_pred.max(1)[1].numpy())
            walking_confusion = confusion_matrix(walking_true.numpy(), walking_pred.max(1)[1].numpy())
            crossing_confusion = confusion_matrix(crossing_true.numpy(), crossing_pred.max(1)[1].numpy())
            confusions[0].append(standing_confusion)
            confusions[1].append(looking_confusion)
            confusions[2].append(walking_confusion)
            confusions[3].append(crossing_confusion)

    metrics = [{} for _ in range(len(losses))]
    for i, (data_confusions, data_losses) in enumerate(zip(confusions, losses)):
        tn, fp, fn, tp = sum(data_confusions).ravel()
        # record metrics
        metrics[i]['d_loss'] = sum(data_losses) / len(data_losses)
        metrics[i]['d_accuracy'] = (tp + tn) / (tn + fp + fn + tp)
        metrics[i]['d_precision'] = tp / (tp + fp)
        metrics[i]['d_recall'] = tp / (tp + fn)
        metrics[i]['d_tn'] = tn
        metrics[i]['d_fp'] = fp
        metrics[i]['d_fn'] = fn
        metrics[i]['d_tp'] = tp

    final_metrics = {}
    for metric in metrics:
        for ind in metric:
            final_metrics[ind] = 0
    for metric in metrics:
        for ind in metric:
            final_metrics[ind] += metric[ind]
    for ind in final_metrics:
        final_metrics[ind] /= len(metrics)

    classifier.train()
    return final_metrics


def cal_l2_losses(pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel,loss_mask):
    g_l2_loss_abs = l2_loss(
        pred_traj_fake, pred_traj_gt, loss_mask, mode='sum'
    )
    g_l2_loss_rel = l2_loss(
        pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum'
    )
    return g_l2_loss_abs, g_l2_loss_rel


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)

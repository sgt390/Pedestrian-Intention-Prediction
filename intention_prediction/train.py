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

from intention_prediction.scripts.models import CNNLSTM1_vgg as CNNLSTM  # , CNNMP
#from intention_prediction.scripts.models import CNNLSTMJAAD2 as CNNLSTM  # , CNNMP
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
parser.add_argument('--min_obs_len', default=8, type=int)
parser.add_argument('--max_obs_len', default=16, type=int)

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
        float_dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    return long_dtype, float_dtype


def main(args):


    debug_op = open("debug_op.txt", "w")
    debug_op.write("\n".join("{!r}: {!r},".format(k, v) for k, v in args.__dict__.items()))
    debug_op.write('\n')
    debug_op.close()
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
    (pedestrian_crops, _, _, _, decision_true, _, _, _, *_) = batch

    losses = {}
    loss = torch.zeros(1).type(torch.cuda.FloatTensor) if torch.cuda.is_available() else torch.zeros(1).type(
        torch.FloatTensor)

    # predict pedestrian decision
    decision_pred = classifier(pedestrian_crops)

    # compute loss
    data_loss = loss_fn(decision_pred, decision_true.cuda()) if torch.cuda.is_available() else loss_fn(decision_pred,
                                                                                                       decision_true.cpu())

    # record loss at current batch and total loss
    losses['data_loss'] = data_loss.item()
    loss += data_loss
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
    data_losses = []
    data_confusions = []
    metrics = {}
    classifier.eval()
    with torch.no_grad():
        for batch in loader:
            # get batch
            (pedestrian_crops, _, _, _, decision_true, *_) = batch

            # predict decision
            decision_pred = classifier(pedestrian_crops) ### todo check if correct (also [0] below)
            # compute loss
            data_loss = loss_fn(decision_pred, decision_true)  #todo decision_true.cpu()
            data_losses.append(data_loss.item())

            # build confusion matrix
            data_confusion = confusion_matrix(decision_true.numpy(), decision_pred.max(1)[1].numpy())
            data_confusion = data_confusion if len(data_confusion) > 1 else np.array([[0, 0], [0, 1]])  # todo remove for real sized inputs
            data_confusions.append(data_confusion)

    tn, fp, fn, tp = sum(data_confusions).ravel()
    # record metrics
    metrics['d_loss'] = sum(data_losses) / len(data_losses)
    metrics['d_accuracy'] = (tp + tn) / (tn + fp + fn + tp)
    metrics['d_precision'] = tp / (tp + fp)
    metrics['d_recall'] = tp / (tp + fn)
    metrics['d_tn'] = tn
    metrics['d_fp'] = fp
    metrics['d_fn'] = fn
    metrics['d_tp'] = tp

    classifier.train()
    return metrics


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

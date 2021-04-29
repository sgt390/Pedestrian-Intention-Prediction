import torch
from intention_prediction.scripts.models import CNNLSTM1_vgg as CNNLSTM
from intention_prediction.scripts.data.loaderJAAD import data_loader
import os
from attrdict import AttrDict
from torch import nn
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib import pyplot as plt
import pprint

loss_fn = nn.CrossEntropyLoss()

PATH = 'model.pt'


def load_model():
    model = CNNLSTM(
        embedding_dim=checkpoint['args']['embedding_dim'],
        h_dim=checkpoint['args']['h_dim'],
        mlp_dim=checkpoint['args']['mlp_dim'],
        dropout=checkpoint['args']['dropout']
    )

    model.load_state_dict(checkpoint['best_state'])
    return model


def load_test(args):
    test_path = os.path.join(checkpoint['args']['dataset'], "val")
    test_dset, test_loader = data_loader(args, test_path, "val")
    return test_loader


if __name__ == '__main__':
    checkpoint = torch.load(PATH)
    pprint.pprint(checkpoint['args'])
    plt.subplot(221)
    plt.title('train accuracy')
    plt.plot(checkpoint['metrics_train']['d_accuracy'])
    plt.subplot(222)
    plt.title('train loss')
    plt.plot(checkpoint['metrics_train']['d_loss'])
    plt.subplot(223)
    plt.title('val accuracy')
    plt.plot(checkpoint['metrics_val']['d_accuracy'])
    plt.subplot(224)
    plt.title('val loss')
    plt.plot(checkpoint['metrics_val']['d_loss'])
    plt.show()

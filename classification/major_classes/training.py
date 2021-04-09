'''
# Training w/ Random Sampling

Sample from Hyperparameter Space randomly N rounds and train model.
All models are saved with training history.
A results.csv summary is created for all models.
'''
import torch
import csv
import random
import torch.nn as nn
import torch.optim as optim
import utils.models as models
import utils.transforms as trans
from utils.datasets import BeeDataset
from utils.train import train_with_params


# constants
N = 50  # number of rounds
num_epochs = 1000
patience = 200
n_classes = 8
db = 'dbs/major_classes'
ds_val = BeeDataset(db, 'val', [trans.resize, trans.totensor])
ds_train = BeeDataset(db, 'train', [trans.augment, trans.totensor])


# Hyperparameter Space
nn_names = ('smallCNN', 'simpleCNN', 'deepCNN')
loss_names = ('xentrop', 'nlll', 'mml')
starting_lrs = (1e-2, 1e-3, 1e-4)
batch_sizes = (200, 400, 800)
dropout_rates = (0.3, 0.7)


# maps
name2loss = dict(
    xentrop=nn.CrossEntropyLoss, nlll=nn.NLLLoss, mml=nn.MultiMarginLoss)
name2nn = dict(
    smallCNN=models.smallCNN, simpleCNN=models.simpleCNN, deepCNN=models.deepCNN)


# results
results_file = 'major_classes/results.csv'
with open(results_file, 'w') as ouf:
    writer = csv.writer(ouf)
    writer.writerow(('Idx', 'Model', 'Loss', 'Start-LR', 'Batch-Size', 'DR', 'Val-Acc'))


for i in range(N):

    # random sampling
    nn_name = random.choice(nn_names)
    loss_name = random.choice(loss_names)
    starting_lr = random.choice(starting_lrs)
    batch_size = random.choice(batch_sizes)
    dropout_rate = round(random.uniform(*dropout_rates), 1)

    model_name = '{i}_{nn}_{loss}-loss_{lr:.0E}-lr_{bs}-bs_{dr}-dr' \
        .format(i=i, nn=nn_name, loss=loss_name, lr=starting_lr, bs=batch_size, dr=dropout_rate)
    print('\n%d: starting training for %s' % (i, model_name))

    # initialize
    model_file = 'major_classes/models/%s.pkl' % model_name
    progress_file = 'major_classes/training/%s.csv' % model_name
    model = name2nn.get(nn_name)(dropout_rate, n_classes)
    criterion = name2loss.get(loss_name)()
    optimizer = optim.Adam(model.parameters(), lr=starting_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [200, 300, 400])

    # train
    best_res = train_with_params(
        ds_train, ds_val, model,
        patience, batch_size, num_epochs,
        scheduler, optimizer, criterion,
        progress_file, model_file)

    # append results
    with open(results_file, 'a') as ouf:
        writer = csv.writer(ouf)
        writer.writerow((i, nn_name, loss_name, starting_lr, batch_size, dropout_rate, best_res))
print('done')

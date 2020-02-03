import torch
import torch.nn as nn
import numpy as np
import glob
import sys
from hyperopt import fmin, tpe, hp, space_eval

from src.JetML.Model import *
from src.JetML.Dataset import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device: ', device)

space = hp.choice('hyper_parameters', [
    {
        'case': sys.argv[1],
        'num_batch': hp.choice('num_batch', [100, 200, 300, 400]),
        'num_layers': hp.quniform('num_layers', 5, 10, 1),
        'learning_rate': hp.uniform('learning_rate', 0.005, 0.01),
        'decay_factor': hp.uniform('decay_factor', 0.5, 0.8),
        'num_epochs':hp.quniform('num_epochs', 5, 20, 5)
    }
])


def train(args):
    # args
    num_batch = int(args['num_batch'])
    num_layers = int(args['num_layers'])
    learning_rate = args['learning_rate']
    decay_factor = args['decay_factor']
    num_epochs = int(args['num_epochs'])
    case = args['case']

    print("Hyper Parameters: ")
    print(args)

    # dataset
    samples = {
        'pythia': './results/ptmin80/pythia_zcut0p5_beta1p5.root',
        'jewel_NR': './results/ptmin80/jewel_NR_zcut0p5_beta1p5.root',
        'jewel_R': './results/ptmin80/jewel_R_zcut0p5_beta1p5.root',
        'hybrid': './results/ptmin80/hybrid_zcut0p5_beta1p5.root'
    }

    print('Pos Class: %s (%s)' % (case, samples[case]))
    print('Neg Class: %s (%s)' % ('pythia', samples['pythia']))

    pos_training = Training_Samples(samples[case], [1., 0.], [0, 30000])
    neg_training = Training_Samples(samples['pythia'], [0., 1.], [0, 30000])
    pos_validation = Training_Samples(samples[case], [1., 0.], [30001, 50000])
    neg_validation = Training_Samples(samples['pythia'], [0., 1.], [ 30001, 50000])

    # training dataloader
    data_loader_training = data.DataLoader(
        data.ConcatDataset([
            pos_training,
            neg_training
        ]),
        batch_size=num_batch, shuffle=True, num_workers=4, drop_last=True, collate_fn=collate_fn_pad
    )
    # validation dataloader
    # batch size 500
    data_loader_validation = data.DataLoader(
        data.ConcatDataset([
            pos_validation,
            neg_validation
        ]),
        batch_size=500, shuffle=True, num_workers=4, drop_last=True, collate_fn=collate_fn_pad
    )

    # model
    model = LSTM(input_size=3, output_size=2, num_layers=num_layers)
    lossFunction = nn.BCELoss()

    # optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # learning rate decay exponentially
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, decay_factor, last_epoch=-1)

    # Training
    for epoch in range(num_epochs):
        for step, (seq, label, length) in enumerate(data_loader_training):
            out, hidden = model(seq)

            # cat the output
            res = out[0][length[0]-1]
            for i in range(1, len(length)):
                res = torch.cat((res,out[i][length[i]-1]), dim=0)
            res = res.view(num_batch, -1)
            res = nn.functional.softmax(res, dim=1)

            loss = lossFunction(res,label)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # if(step%10==0):
            #     print('Eopch: [{}/{}], Step: {}, Loss: {:.2f}'.format(epoch+1, num_epochs, step, loss.item()))

    # validation
    # batch_size 500
    loss_total = 0
    corr_total = 0
    n_total = 0
    for step, (seq, label, length) in enumerate(data_loader_validation):
        out, hidden = model(seq)
        # cat the output
        res = out[0][length[0]-1]
        for i in range(1, len(length)):
            res = torch.cat((res,out[i][length[i]-1]), dim=0)
        res = res.view(500, -1)
        res = nn.functional.softmax(res, dim=1)

        loss = lossFunction(res,label)
        loss_total = loss_total + loss.item()

        corr = (res.argmax(dim=1)==label.argmax(dim=1)).sum().item()
        corr_total = corr_total + corr
        n_total = n_total + 500

    accuracy = 100 * corr_total / n_total
    print('Validation Loss: {:.4f}, Accuracy: {:.2f} %'.format(loss_total, accuracy))
    return loss_total

best = fmin(train, space, algo=tpe.suggest, max_evals=20)
print(space_eval(space, best))

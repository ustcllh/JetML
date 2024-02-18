import torch
import torch.nn as nn
import numpy as np
import glob
import time
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, space_eval, STATUS_OK, Trials
import json

from src.JetML.Model import *
from src.JetML.Dataset import *

start_time = time.time()

# using CPU/GPU
cpu = torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device: ', device)

# flags
save_intermediate_results = False
save_loss_plots = False

# hyper tuning space
max_evals = 5
space = hp.choice('hyper_parameters',[
    {
        'num_batch': hp.quniform('num_batch', 10000, 20000, 2000),
        'num_epochs': hp.quniform('num_epochs', 40, 50, 5),
        'num_layers': hp.quniform('num_layers', 2, 4, 1),
        'hidden_size0': hp.quniform('hidden_size0', 8, 20, 2),
        'hidden_size1': hp.quniform('hidden_size1', 4, 8, 2),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.05),
        'decay_factor': hp.uniform('decay_factor', 0.9, 0.99),
        'loss_func' : hp.choice('loss_func', ['mse']),
    }
])

# cuts
jetpt_cut = 200
rg_cut = 0.1
zg_cut = 0.1

# samples
jewel_training = Samples('./data/Training/jewel_R_csejet_pt200_rg0p1_mult7000.root', 'csejet', [1., 0.], [0, 100000], zg_cut=zg_cut, rg_cut=rg_cut, jetpt_cut=jetpt_cut)
pythia_training = Samples('./data/Training/pythia_csejet_pt200_rg0p1_mult7000.root', 'csejet', [0., 1.], [0, 100000], zg_cut=zg_cut, rg_cut=rg_cut, jetpt_cut=jetpt_cut)

jewel_validation = Samples('./data/Validation/jewel_R_csejet_pt200_rg0p1_mult7000.root', 'csejet', [1., 0.], [0, 100000], zg_cut=zg_cut, rg_cut=rg_cut, jetpt_cut=jetpt_cut)
pythia_validation = Samples('./data/Training/pythia_csejet_pt200_rg0p1_mult7000.root', 'csejet', [0., 1.], [0, 100000], zg_cut=zg_cut, rg_cut=rg_cut, jetpt_cut=jetpt_cut)

print('# of Jets (training): %d jewel jets, %d pythia jets' % (jewel_training.len, pythia_training.len))
print('# of Jets (vlidation): %d jewel jets, %d pythia jets' % (jewel_validation.len, pythia_validation.len))
print('jet pt cut: %d GeV' % jetpt_cut)
print('zg cut: %f' % zg_cut)
print('rg cut: %f' % rg_cut)

# weighted mse loss
def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)/torch.sum(weight)

# weighted bce loss
def weighted_bce_loss(input, target, weight):
    return torch.nn.functional.binary_cross_entropy(input, target, weight, reduction='sum')/torch.sum(weight) 

itrial = 0
nattempts = 3
index_trial = []
loss_trial = []
def train(args):
    global itrial
    itrial = itrial + 1
    # args
    num_batch = int(args['num_batch'])
    num_epochs = int(args['num_epochs'])
    num_layers = int(args['num_layers'])
    hidden_size0 = int(args['hidden_size0'])
    hidden_size1 = int(args['hidden_size1'])
    learning_rate = args['learning_rate']
    decay_factor = args['decay_factor']
    loss_func = args['loss_func']

    print("Hyper Parameters")
    print(args)

    data_loader_training = data.DataLoader(
        data.ConcatDataset([
            jewel_training,
            pythia_training
        ]),
        batch_size=num_batch, shuffle=True, num_workers=8, drop_last=True, collate_fn=collate_fn_pad
    )

    data_loader_validation = data.DataLoader(
        data.ConcatDataset([
           jewel_validation,
           pythia_validation
        ]),
        batch_size=num_batch, shuffle=True, num_workers=8, drop_last=True, collate_fn=collate_fn_pad
    )

    loss_minimum = 2
    model_best = None

    for i in range(nattempts):
        # LSTM + FC + RELU + FC
        model = LSTM_FC(input_size=4, hidden_size=[hidden_size0, hidden_size1], num_layers=num_layers, batch_size=num_batch, device=device)

        # optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # learning rate decay exponentially
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, decay_factor, last_epoch=-1)
    
        # training
        step_training = []
        loss_training = []
        istep = 0
        for epoch in range(num_epochs):
            scheduler.step()
            for step, (seq, weight, label, length) in enumerate(data_loader_training):
                seq = seq.to(device)
                out = model(seq)
                out = out.to(cpu)
                res = nn.functional.softmax(out, dim=1)

                loss = 0
                if loss_func == 'mse':
                    loss = weighted_mse_loss(res, label, weight)
                if loss_func == 'bce':
                    loss = weighted_bce_loss(res, label, weight)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                istep = istep+1
                step_training.append(istep)
                loss_training.append(float(loss))

                if (step+1) % 10 == 0:
                    print('Trial: {}, Attempt: {}, Eopch: [{}/{}], LR: {}, Step: {}, Loss:{:.4f}'.format(itrial, i, epoch, num_epochs, scheduler.get_lr(), step+1, float(loss)))


        # validation
        loss_sum = 0.
        weight_sum = 0.
        model.eval()
        for step, (seq, weight, label, length) in enumerate(data_loader_validation):
            seq = seq.to(device)
            with torch.no_grad():
                out = model(seq)
                out = out.to(cpu)
                res = nn.functional.softmax(out, dim=1)
                if loss_func == 'bce':
                    loss_sum += float(torch.nn.functional.binary_cross_entropy(res, label, weight, reduction='sum'))
                elif loss_func == 'mse':
                    loss_sum += float(torch.sum(weight * (res - label) ** 2))
                weight_sum += torch.sum(weight)
        model.train()
        

        result = loss_sum/weight_sum
        if result.item()<loss_minimum:
            loss_minimum = result.item()
            model_best = model

        # loss vs step
        if save_loss_plots:
            plt.clf()
            plt.plot(step_training, loss_training)
            plt.xlabel('Step Index')
            plt.ylabel('Loss')
            if loss_func == 'mse':
                plt.ylim(0.3, 0.6)
                plt.savefig('./loss/MSE_{}_{}.png'.format(itrial, i))
            if loss_func == 'bce':
                plt.ylim(0.8, 1.5)
                plt.savefig('./loss/BCE_{}_{}.png'.format(itrial, i))

        if save_intermediate_results:
            prefix = 'zcut0p1_beta0_csejet_' + loss_func + '_mult7000_pt' + str(jetpt_cut) +'_itrial_' + str(itrial) + '_' + str(i)
            model_path = './model/' + prefix + '.pt'
            torch.save(model.state_dict(), model_path)
            json_path = './model/' + prefix + '.json'
            json_args = args
            json_args['loss'] = result.item()
            with open(json_path, 'w') as json_file:
                json.dump(json_args, json_file, indent=4)


        print('Trial: {}, Attempt: {}, Validation Loss {}'.format(itrial, i, result.item()))


    global index_trial
    global loss_trial
    index_trial.append(itrial)
    loss_trial.append(loss_minimum)
    if loss_func == 'bce':
        print('Validation (BCE LOSS): %.4f' % loss_minimum)
    if loss_func == 'mse':
        print('Validation (MSE LOSS): %.4f' % loss_minimum)
    return {'loss': loss_minimum, 'status': STATUS_OK, 'model': model_best, 'loss_func':loss_func}



trials = Trials()
best = fmin(train, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
print(space_eval(space,best))

def getBestModelfromTrials(trials):
    valid_trial_list = [trial for trial in trials if STATUS_OK == trial['result']['status']]
    losses = [ float(trial['result']['loss']) for trial in valid_trial_list]
    index_having_minumum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss]
    return best_trial_obj['result']['model'], best_trial_obj['result']['loss_func']

model_select, loss_func = getBestModelfromTrials(trials)
# save model parameters
import json
prefix = 'zcut0p1_beta0_csejet_' + loss_func + '_mult7000_pt' + str(jetpt_cut) + '_best'
model_path = './model/' + prefix + '.pt'
torch.save(model_select.state_dict(), model_path)
json_path = './model/' + prefix + '.json'
with open(json_path, 'w') as json_file:
    json.dump(space_eval(space,best), json_file, indent=4)
print("--- Training: %s seconds ---" % (time.time() - start_time))


# loss vs step
loss_min, index_min = min((loss, index) for (index,loss) in enumerate(loss_trial))

plt.clf()
plt.plot(index_trial, loss_trial)
plt.plot([index_min+1], [loss_min], 'ro')
plt.xlabel('Trial Index')
plt.ylabel('Loss')
if loss_func == 'mse':
    plt.ylim(0.3, 0.6)
    plt.savefig('./loss/MSE_trial.png')
if loss_func == 'bce':
    plt.ylim(0.8, 1.5)
    plt.savefig('./loss/BCE_trial.png')

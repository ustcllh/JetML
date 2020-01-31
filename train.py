import torch
import torch.nn as nn
import numpy as np
import glob
import time

from src.JetML.Model import *
from src.JetML.Dataset import *

start_time = time.time()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device: ', device)

njet = 40000
prefix = 'jewel_R_zcut0p5_beta1p5'
jewel_training = Training_Samples('./results/ptmin80/jewel_R_zcut0p5_beta1p5.root', [1., 0.], njet)
pythia_training = Training_Samples('./results/ptmin80/pythia_zcut0p5_beta1p5.root', [0., 1.], njet)

print('# of Jets (training): %d jewel jets, %d pythia jets' % (jewel_training.len, pythia_training.len))

num_batch = 100
data_loader_training = data.DataLoader(
    data.ConcatDataset([
        jewel_training,
        pythia_training
    ]),
    batch_size=num_batch, shuffle=True, num_workers=4, drop_last=True, collate_fn=collate_fn_pad
)

# iter = iter(data_loader_training)
# batch = iter.next()
# print(batch)

model = LSTM(batch_size=num_batch, input_size=3, output_size=2, num_layers=5)
lossFunction = nn.BCELoss()

# optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# learning rate decay exponentially
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.8, last_epoch=-1)

num_epochs = 20
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

        if (step+1) % 10 == 0:
            print('Epoch:', epoch,'LR:', scheduler.get_lr())
            print('Eopch: [{}/{}], Step: {}, Loss:{:.2f}'.format(epoch, num_epochs, step+1, loss))

            corr = (res.argmax(dim=1)==label.argmax(dim=1)).sum().item()
            accuracy = 100.00 * corr / num_batch
            print('Accuracy[Corr/Total]: [{}/{}] = {:.2f} %' .format(corr, num_batch, accuracy))

    scheduler.step()

# save model parameters
model_path = './model/' + prefix + '.pt'
torch.save(model.state_dict(), model_path)

print("--- Training: %s seconds ---" % (time.time() - start_time))

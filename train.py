import torch
import torch.nn as nn
import numpy as np
import glob
import time
import matplotlib.pyplot as plt

from src.JetML.Model import *
from src.JetML.Dataset import *

start_time = time.time()

cpu = torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print('Using device: ', device)

prefix = 'jewel_R'
jewel_training = Training_Samples('./results/ptmin130/jewel_R.root', [1., 0.], [0, 50000])
pythia_training = Training_Samples('./results/ptmin130/pythia.root', [0., 1.], [0, 50000])

print('# of Jets (training): %d jewel jets, %d pythia jets' % (jewel_training.len, pythia_training.len))

lo = []
st = []
num_batch = 1000
data_loader_training = data.DataLoader(
    data.ConcatDataset([
        jewel_training,
        pythia_training
    ]),
    batch_size=num_batch, shuffle=True, num_workers=2, drop_last=True, collate_fn=collate_fn_pad
)

# iter = iter(data_loader_training)
# batch = iter.next()
# print(batch)

model = LSTM(input_size=4, output_size=2, num_layers=5, device=device)

# optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# learning rate decay exponentially
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9, last_epoch=-1)

num_epochs = 3
for epoch in range(num_epochs):
    for step, (seq, weight, label, length) in enumerate(data_loader_training):
        seq = seq.to(device)

        out, hidden = model(seq)

        out = out.to(cpu)

        # cat the output
        res = out[0][length[0]-1]
        for i in range(1, len(length)):
            res = torch.cat((res,out[i][length[i]-1]), dim=0)
        res = res.view(num_batch, -1)
        res = nn.functional.softmax(res, dim=1)

        # lossFunction = nn.BCELoss(weight=weight)
        lossFunction = nn.BCELoss()
        loss = lossFunction(res,label)
        lo.append(loss.item())
        st.append(len(lo))
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
# model_path = './model/' + prefix + '.pt'
# torch.save(model.state_dict(), model_path)

print(st)
print(lo)

plt.plot(st, lo)
plt.show()
plt.savefig('loss_step.png')

print("--- Training: %s seconds ---" % (time.time() - start_time))

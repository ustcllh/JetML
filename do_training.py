import torch
import torch.nn as nn
import numpy as np
import glob
from src.jetml.Model import *
from src.jetml.Dataset import *
import time
start_time = time.time()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device: ', device)


mask = [True, True, False, True, False, False]
dim = np.sum(mask)
jewel_training = Training_Samples('./data/cambridge/training/jewel', [1., 0.], mask)
pythia_training = Training_Samples('./data/cambridge/training/pythia', [0., 1.], mask)

print('# of Jets (training): %d jewel jets, %d pythia jets' % (jewel_training.len, pythia_training.len))
print('Mask: ', mask)
print('Dim: %d' % dim)


data_loader_training = data.DataLoader(
    data.ConcatDataset([
        jewel_training,
        pythia_training
    ]),
    batch_size=100, shuffle=True, num_workers=4, drop_last=True, collate_fn=collate_fn_pad
)

# iter = iter(data_loader)
# batch = iter.next()
# print(batch)

num_batch = 100

model = lstm_network(dim)
lossFunction = nn.BCELoss()

# optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# learning rate decay exponentially
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.8, last_epoch=-1)

num_epochs = 10

for epoch in range(num_epochs):
    scheduler.step()
    for step, (seq, label, length) in enumerate(data_loader_training):
        hidden = model.hidden_init()
        out, hidden = model(seq, hidden)

        # cat the output before padding
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

        # model_path = './model/lstm_11000_%d.pt' % epoch
        # torch.save(model.state_dict(), model_path)


# save model parameters
model_path = './model/lstm_11010.pt'
torch.save(model.state_dict(), model_path)

print("--- Training: %s seconds ---" % (time.time() - start_time))

# validation

jewel_validation = Training_Samples('./data/cambridge/validation/jewel', [1., 0.], mask)

pythia_validation = Training_Samples('./data/cambridge/validation/pythia', [0., 1.], mask)

print('# of Jets (validation): %d jewel jets, %d pythia jets' % (jewel_validation.len, pythia_validation.len))


data_loader_validation = data.DataLoader(
    data.ConcatDataset([
        jewel_validation,
        pythia_validation
    ]),
    batch_size=100, shuffle=True, num_workers=4, drop_last=True, collate_fn=collate_fn_pad
)


corr = 0
total = 0


for step, (seq, label, length) in enumerate(data_loader_validation):

    hidden = model.hidden_init()
    out, hidden = model(seq, hidden)

    res = out[0][length[0]-1]
    for i in range(1, len(length)):
        res = torch.cat((res,out[i][length[i]-1]), dim=0)
    res = res.view(num_batch, -1)
    res = nn.functional.softmax(res, dim=1)

    corr += (res.argmax(dim=1)==label.argmax(dim=1)).sum().item()
    total += num_batch


accuracy = 100.00 * corr / total
print('Validation Accuracy[Corr/Total]: [{}/{}] = {:.2f} %' .format(corr, total, accuracy))

print("--- %s seconds ---" % (time.time() - start_time))

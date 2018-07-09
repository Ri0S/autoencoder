import configs
import dataloader
import pickle
from tqdm import tqdm
import model
import torch
import sampling
from torch import optim
from torch import nn

config = configs.get_config()

with open('./data/i2c', 'rb') as f:
    config.vocab_size = len(pickle.load(f))

autoencoder = model.autoEncoder(config).to(config.device)

trainLoader = dataloader.get_dataloader(config.batch_size, True)

optimizer = optim.Adam(autoencoder.parameters(), lr=config.learning_rate)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)

for i in range(config.n_epoch):
    epoch_loss = 0
    for num, data in enumerate(tqdm(trainLoader)):
        optimizer.zero_grad()

        outs = autoencoder(data)
        loss = loss_fn(outs.view(-1, config.vocab_size), data[1][:, 1:].contiguous().view(-1))
        epoch_loss += loss
        loss.backward()
        optimizer.step()
    print('epoch', i, 'sample')
    for idx in range(10):
        sampling(data[0][idx], torch.topk[outs[idx], 1, 1][1])
    print('loss:', epoch_loss / num)
    torch.save(autoencoder.state_dict(), config.path + str(i))

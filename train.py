from torch.nn import L1Loss, BCELoss
from torch.optim import Adam, lr_scheduler
from discriminator import Discriminator
from generator import Generator
from dataloaders import get_dataloader, cvt_back
from tqdm import tqdm
import cv2
import os
import torch

epoch_num = 0
lambd = 100
smoothing = 0.9
def train_epoch(train_dl, val_dl, gen, disc, loss_gen, loss_disc, gen_optim, disc_optim, gen_sched, disc_sched):
    global epoch_num

    gen_adv_loss = 0
    gen_l1_loss = 0
    disc_real_loss = 0
    disc_fake_loss = 0

    # train
    for X in tqdm(train_dl):

        x_l = X[:, [0], :, :]
        x_ab = X[:, 1:3, :, :]

        target_ones = torch.ones(X.size(0), 1, 1, 1).to('cuda')
        target_zeros = torch.zeros(X.size(0), 1, 1, 1).to('cuda')

        # training the generator
        gen_optim.zero_grad()
        fake = gen(x_l)

        disc_input = torch.cat([x_l, fake], dim=1)
        disc_pred = disc(disc_input)

        adv_loss = loss_disc(disc_pred, target_ones)
        l1_loss = loss_gen(fake, x_ab)
        gen_loss = adv_loss + lambd*l1_loss

        gen_adv_loss += adv_loss.item()
        gen_l1_loss += l1_loss.item()

        gen_loss.backward()
        gen_optim.step()
        gen_sched.step(gen_loss)

        # training the discriminator
        disc_optim.zero_grad()
        disc_pred_real = disc(X)
        disc_pred_fake = disc(torch.cat([x_l, fake.detach()], dim=1))

        real_loss = loss_disc(disc_pred_real, target_ones*smoothing)
        fake_loss = loss_disc(disc_pred_fake, target_zeros)
        disc_loss = real_loss + fake_loss

        disc_real_loss += real_loss.item()
        disc_fake_loss += fake_loss.item()

        disc_loss.backward()
        disc_optim.step()
        disc_sched.step(disc_loss)

        # batch_num += 1
        # if batch_num % 50 == 0:
        #     print(f'batch {batch_num} LOSS: {loss}')

    gen_adv_loss /= len(train_dl)
    gen_l1_loss /= len(train_dl)
    disc_real_loss /= len(train_dl)
    disc_fake_loss /= len(train_dl)
    print(gen_adv_loss, '\t', gen_l1_loss, '\t', disc_real_loss, '\t', disc_fake_loss)

    for i, x in enumerate(fake):
        processed = cvt_back(x.cpu().detach())
        # cv2.imshow('hello', processed)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join('output', str(epoch_num)+'_'+str(i)+'.jpg'), processed)
        # write_jpeg(processed, str(epoch_num)+'_'+str(i)+'.jpg')
    torch.save(gen.state_dict(), os.path.join('models', 'gen_' + str(epoch_num) + '.pt'))
    torch.save(disc.state_dict(), os.path.join('models', 'disc_' + str(epoch_num) + '.pt'))
    epoch_num += 1


def train():
    gen = Generator().to('cuda')
    disc = Discriminator().to('cuda')

    # gen.load_state_dict(torch.load('gan2/gen_58.pt'))
    # disc.load_state_dict(torch.load('gan2/disc_58.pt'))

    loss_gen = L1Loss()
    loss_disc = BCELoss()
    gen_optim = Adam(gen.parameters(), lr=2e-4)
    disc_optim = Adam(disc.parameters(), lr=2e-4)
    gen_sched = lr_scheduler.ReduceLROnPlateau(gen_optim, patience=5)
    disc_sched = lr_scheduler.ReduceLROnPlateau(disc_optim, patience=5)
    train_dl = get_dataloader('/home/vuk/Documents/Places365/train')
    val_dl = get_dataloader('/home/vuk/Documents/Places365/val')

    epochs = 100
    for _ in range(epochs):
        train_epoch(train_dl, val_dl, gen, disc, loss_gen, loss_disc, gen_optim, disc_optim, gen_sched, disc_sched)


if __name__ == '__main__':
    train()
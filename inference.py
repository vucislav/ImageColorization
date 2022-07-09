import cv2
import os
import numpy as np
import torch
from dataloaders import cvt_back, get_dataloader
from generator import Generator


if __name__ == '__main__':
    gen = Generator().cuda()
    gen.load_state_dict(torch.load('gan1/gen_29.pt'))

    dl = get_dataloader('/home/vuk/Documents/Places365/train')

    iters = 0
    max_iters = 5
    out_dir = 'inference'
    os.makedirs(out_dir, exist_ok=True)
    for X, y in dl:
        if iters >= max_iters:
            break
        res = gen(X)
        for i, (bw, real, fake) in enumerate(zip(X, y, res)):
            fake = cvt_back(fake.cpu().detach())
            real = cvt_back(real.cpu().detach())
            combined = np.hstack((real, fake))
            cv2.imwrite(os.path.join(out_dir, f'{iters}_{i}.png'), combined)
        iters += 1

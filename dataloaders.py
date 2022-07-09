import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor

class ColorizationDataset(Dataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.ds_path = dataset_path
        self.file_names = os.listdir(dataset_path)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        full_path = os.path.join(self.ds_path, self.file_names[idx])
        img = cv2.imread(full_path)

        img = cvt(img)
        
        return img


def get_dataloader(dataset_path):
    ds = ColorizationDataset(dataset_path)
    return DataLoader(ds, batch_size=4, shuffle=True, drop_last=True, num_workers=0)

def cvt(x):
    img = x.astype(np.float32)/255.0          

    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # bw = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), axis=2)

    img[:, :, 0] = img[:, :, 0]/50 - 1
    img[:, :, 1] = img[:, :, 1]/127
    img[:, :, 2] = img[:, :, 2]/127

    img = to_tensor(img).to('cuda')
    # bw = to_tensor(bw).to('cuda')

    return img


def cvt_back(x):
    img = x.numpy().transpose((1, 2, 0))

    img[:, :, 0] = (1+img[:, :, 0])*50
    img[:, :, 1] = img[:, :, 1]*127
    img[:, :, 2] = img[:, :, 2]*127

    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    img = (255*img).astype(np.uint8)
    return img

if __name__ == '__main__':
    dl = get_dataloader('val_256')
    for _ in dl:
        pass
    
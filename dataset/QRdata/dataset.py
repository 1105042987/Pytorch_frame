from PIL import Image

# For network and training
import torch
import torch.nn as nn
import torch.nn.functional as F

# For dataset
import torchvision
import torchvision.transforms as T

class __Dataset(torch.utils.data.__Dataset):
    def __init__(self, dataType, lens, img_transforms=None, **kwargs):
        super(__Dataset, self).__init__()
        if dataType.lower() == 'train':
            self.img_dir = './train/img/'
            self.lab_dir = './train/label/'
        elif dataType.lower() == 'test':
            self.img_dir = './test/img/'
            self.lab_dir = './test/label/'
        self.transforms = img_transforms
        subnum = 0
        while isExist(self.img_dir+'{}.png'.format(subnum)):
            subnum += 1
        self.lens = subnum
        if self.lens == 0:
            raise('No data')

    def __getitem__(self, index):
        name = '%d' % index
        img = Image.open(self.img_dir+name+'.png')
        if self.transforms is None:
            raise('No Transform')
        img = self.transforms(img).float()
        label = torch.from_numpy(np.load(self.lab_dir+name+'.npz')['data']).float()
        label[label == 2] = 0.5
        return img, label

    def __len__(self):
        return self.lens


# Define transform for image dataset
def loadData(name,Num,batch):
    if name == "train":
        transform_train = T.Compose([
            # T.RandomHorizontalFlip(), # 随机的旋转给定的PIL，数据增强
            T.ToTensor(),  # Important!
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = __Dataset('train', 30, transform_train)
        return torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=2)

    elif name == "test":
        transform_test = T.Compose([
            T.ToTensor(),  # Important!
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = __Dataset('test', 30, transform_test)
        return torch.utils.data.DataLoader(testset, batch_size=batch, shuffle=False, num_workers=2)

from frame.frame import weak_ErrorRate,weak_GetPredictAnswer,onehot2num

class ErrorRate(weak_ErrorRate):
    def __init__(self):
        super(ErrorRate, self).__init__()

    def add(self, out, tar):
        import torch
        self.total += len(tar)
        outputs = torch.max(out, 1)[1]
        self.error += torch.sum(outputs != tar).float()


class AnsGet(weak_GetPredictAnswer):
    def __init__(self):
        super(AnsGet, self).__init__()

    def add(self, outputs, index):
        from torch import cat
        if self.data is None:
            self.data = cat((index, onehot2num(outputs).to('cpu')), 1)
        else:
            outputs = cat((index, onehot2num(outputs).to('cpu')), 1)
            self.data = cat((self.data, outputs))

    def save(self, name):
        import pandas as pd
        ans = pd.DataFrame(np.array(self.data), columns=['ID', 'Category'])
        ans.to_csv('./output/{}.csv'.format(name), header=True, index=False)


if __name__ == '__main__':
    pass

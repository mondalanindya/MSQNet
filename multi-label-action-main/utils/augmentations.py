from datasets.transforms_ss import *
from RandAugment import RandAugment

class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

def get_augmentation(mode='train'):
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    input_size = 224
    scale_size = 256
    if mode == 'train':
        unique = T.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                            GroupRandomHorizontalFlip(True),
                            GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                            GroupRandomGrayscale(p=0.2),
                            GroupGaussianBlur(p=0.0),
                            GroupSolarization(p=0.0)]
                            )
    else:
        unique = T.Compose([GroupScale(scale_size),
                            GroupCenterCrop(input_size)])

    common = T.Compose([Stack(roll=False),
                        ToTorchFormatTensor(div=True),
                        GroupNormalize(input_mean, input_std)])
    return T.Compose([unique, common])

def randAugment(transform_train,config):
    print('Using RandAugment!')
    transform_train.transforms.insert(0, GroupTransform(RandAugment(config.data.randaug.N, config.data.randaug.M)))
    return transform_train
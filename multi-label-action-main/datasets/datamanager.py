import torch
import random
import itertools
import numpy as np
from .transforms_ss import *
from torchvision.transforms import Compose
from catalyst.data.sampler import DistributedSamplerWrapper
from datasets.datasets import AnimalKingdom, Charades, Hockey, Thumos14, Volleyball
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler
# from pytorchvideo.transforms import ApplyTransformToKey, create_video_transform


class LimitDataset(Dataset):
    """
    To ensure a constant number of samples are retrieved from the dataset we use this
    LimitDataset wrapper. This is necessary because several of the underlying videos
    may be corrupted while fetching or decoding, however, we always want the same
    number of steps per epoch.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(dataset), 2)
        )

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos
    

class DataManager():
    def __init__(self, args, path):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.path = path
        self.dataset = args.dataset
        self.total_length = args.total_length
        self.test_part = args.test_part
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.distributed = args.distributed

    def _check(self,):
        datasets_list = ["animalkingdom", "ava", "charades", "hockey", "thumos14", "volleyball"]
        if(self.dataset not in datasets_list):
            raise Exception("[ERROR] The dataset " + str(self.dataset) + " is not supported!")

    def get_num_classes(self,):
        self._check()
        if(self.dataset == "animalkingdom"): return 140
        elif(self.dataset == "ava"): return 80
        elif(self.dataset == "charades"): return 157
        elif(self.dataset == "hockey"): return 12
        elif(self.dataset == "thumos14"): return 20
        elif(self.dataset == "volleyball"): return 9

    def get_act_dict(self,):
        self._check()
        animalkingdom_dict = {'Abseiling': 0, 'Attacking': 1, 'Attending': 2, 'Barking': 3, 'Being carried': 4, 'Being carried in mouth': 5, 'Being dragged': 6, 'Being eaten': 7, 'Biting': 8, 'Building nest': 9, 'Calling': 10, 'Camouflaging': 11, 'Carrying': 12, 'Carrying in mouth': 13, 'Chasing': 14, 'Chirping': 15, 'Climbing': 16, 'Coiling': 17, 'Competing for dominance': 18, 'Dancing': 19, 'Dancing on water': 20, 'Dead': 21, 'Defecating': 22, 'Defensive rearing': 23, 'Detaching as a parasite': 24, 'Digging': 25, 'Displaying defensive pose': 26, 'Disturbing another animal': 27, 'Diving': 28, 'Doing a back kick': 29, 'Doing a backward tilt': 30, 'Doing a chin dip': 31, 'Doing a face dip': 32, 'Doing a neck raise': 33, 'Doing a side tilt': 34, 'Doing push up': 35, 'Doing somersault': 36, 'Drifting': 37, 'Drinking': 38, 'Dying': 39, 'Eating': 40, 'Entering its nest': 41, 'Escaping': 42, 'Exiting cocoon': 43, 'Exiting nest': 44, 'Exploring': 45, 'Falling': 46, 'Fighting': 47, 'Flapping': 48, 'Flapping tail': 49, 'Flapping its ears': 50, 'Fleeing': 51, 'Flying': 52, 'Gasping for air': 53, 'Getting bullied': 54, 'Giving birth': 55, 'Giving off light': 56, 'Gliding': 57, 'Grooming': 58, 'Hanging': 59, 'Hatching': 60, 'Having a flehmen response': 61, 'Hissing': 62, 'Holding hands': 63, 'Hopping': 64, 'Hugging': 65, 'Immobilized': 66, 'Jumping': 67, 'Keeping still': 68, 'Landing': 69, 'Lying down': 70, 'Laying eggs': 71, 'Leaning': 72, 'Licking': 73, 'Lying on its side': 74, 'Lying on top': 75, 'Manipulating object': 76, 'Molting': 77, 'Moving': 78, 'Panting': 79, 'Pecking': 80, 'Performing sexual display': 81, 'Performing allo-grooming': 82, 'Performing allo-preening': 83, 'Performing copulatory mounting': 84, 'Performing sexual exploration': 85, 'Performing sexual pursuit': 86, 'Playing': 87, 'Playing dead': 88, 'Pounding': 89, 'Preening': 90, 'Preying': 91, 'Puffing its throat': 92, 'Pulling': 93, 'Rattling': 94, 'Resting': 95, 'Retaliating': 96, 'Retreating': 97, 'Rolling': 98, 'Rubbing its head': 99, 'Running': 100, 'Running on water': 101, 'Sensing': 102, 'Shaking': 103, 'Shaking head': 104, 'Sharing food': 105, 'Showing affection': 106, 'Sinking': 107, 'Sitting': 108, 'Sleeping': 109, 'Sleeping in its nest': 110, 'Spitting': 111, 'Spitting venom': 112, 'Spreading': 113, 'Spreading wings': 114, 'Squatting': 115, 'Standing': 116, 'Standing in alert': 117, 'Startled': 118, 'Stinging': 119, 'Struggling': 120, 'Surfacing': 121, 'Swaying': 122, 'Swimming': 123, 'Swimming in circles': 124, 'Swinging': 125, 'Tail swishing': 126, 'Trapped': 127, 'Turning around': 128, 'Undergoing chrysalis': 129, 'Unmounting': 130, 'Unrolling': 131, 'Urinating': 132, 'Walking': 133, 'Walking on water': 134, 'Washing': 135, 'Waving': 136, 'Wrapping itself around prey': 137, 'Wrapping prey': 138, 'Yawning': 139}
        charades_dict = {'Holding some clothes': 0, 'Putting clothes somewhere': 1, 'Taking some clothes from somewhere': 2, 'Throwing clothes somewhere': 3, 'Tidying some clothes': 4, 'Washing some clothes': 5, 'Closing a door': 6, 'Fixing a door': 7, 'Opening a door': 8, 'Putting something on a table': 9, 'Sitting on a table': 10, 'Sitting at a table': 11, 'Tidying up a table': 12, 'Washing a table': 13, 'Working at a table': 14, 'Holding a phone/camera': 15, 'Playing with a phone/camera': 16, 'Putting a phone/camera somewhere': 17, 'Taking a phone/camera from somewhere': 18, 'Talking on a phone/camera': 19, 'Holding a bag': 20, 'Opening a bag': 21, 'Putting a bag somewhere': 22, 'Taking a bag from somewhere': 23, 'Throwing a bag somewhere': 24, 'Closing a book': 25, 'Holding a book': 26, 'Opening a book': 27, 'Putting a book somewhere': 28, 'Smiling at a book': 29, 'Taking a book from somewhere': 30, 'Throwing a book somewhere': 31, 'Watching/Reading/Looking at a book': 32, 'Holding a towel/s': 33, 'Putting a towel/s somewhere': 34, 'Taking a towel/s from somewhere': 35, 'Throwing a towel/s somewhere': 36, 'Tidying up a towel/s': 37, 'Washing something with a towel': 38, 'Closing a box': 39, 'Holding a box': 40, 'Opening a box': 41, 'Putting a box somewhere': 42, 'Taking a box from somewhere': 43, 'Taking something from a box': 44, 'Throwing a box somewhere': 45, 'Closing a laptop': 46, 'Holding a laptop': 47, 'Opening a laptop': 48, 'Putting a laptop somewhere': 49, 'Taking a laptop from somewhere': 50, 'Watching a laptop or something on a laptop': 51, 'Working/Playing on a laptop': 52, 'Holding a shoe/shoes': 53, 'Putting shoes somewhere': 54, 'Putting on shoe/shoes': 55, 'Taking shoes from somewhere': 56, 'Taking off some shoes': 57, 'Throwing shoes somewhere': 58, 'Sitting in a chair': 59, 'Standing on a chair': 60, 'Holding some food': 61, 'Putting some food somewhere': 62, 'Taking food from somewhere': 63, 'Throwing food somewhere': 64, 'Eating a sandwich': 65, 'Making a sandwich': 66, 'Holding a sandwich': 67, 'Putting a sandwich somewhere': 68, 'Taking a sandwich from somewhere': 69, 'Holding a blanket': 70, 'Putting a blanket somewhere': 71, 'Snuggling with a blanket': 72, 'Taking a blanket from somewhere': 73, 'Throwing a blanket somewhere': 74, 'Tidying up a blanket/s': 75, 'Holding a pillow': 76, 'Putting a pillow somewhere': 77, 'Snuggling with a pillow': 78, 'Taking a pillow from somewhere': 79, 'Throwing a pillow somewhere': 80, 'Putting something on a shelf': 81, 'Tidying a shelf or something on a shelf': 82, 'Reaching for and grabbing a picture': 83, 'Holding a picture': 84, 'Laughing at a picture': 85, 'Putting a picture somewhere': 86, 'Taking a picture of something': 87, 'Watching/looking at a picture': 88, 'Closing a window': 89, 'Opening a window': 90, 'Washing a window': 91, 'Watching/Looking outside of a window': 92, 'Holding a mirror': 93, 'Smiling in a mirror': 94, 'Washing a mirror': 95, 'Watching something/someone/themselves in a mirror': 96, 'Walking through a doorway': 97, 'Holding a broom': 98, 'Putting a broom somewhere': 99, 'Taking a broom from somewhere': 100, 'Throwing a broom somewhere': 101, 'Tidying up with a broom': 102, 'Fixing a light': 103, 'Turning on a light': 104, 'Turning off a light': 105, 'Drinking from a cup/glass/bottle': 106, 'Holding a cup/glass/bottle of something': 107, 'Pouring something into a cup/glass/bottle': 108, 'Putting a cup/glass/bottle somewhere': 109, 'Taking a cup/glass/bottle from somewhere': 110, 'Washing a cup/glass/bottle': 111, 'Closing a closet/cabinet': 112, 'Opening a closet/cabinet': 113, 'Tidying up a closet/cabinet': 114, 'Someone is holding a paper/notebook': 115, 'Putting their paper/notebook somewhere': 116, 'Taking paper/notebook from somewhere': 117, 'Holding a dish': 118, 'Putting a dish/es somewhere': 119, 'Taking a dish/es from somewhere': 120, 'Wash a dish/dishes': 121, 'Lying on a sofa/couch': 122, 'Sitting on sofa/couch': 123, 'Lying on the floor': 124, 'Sitting on the floor': 125, 'Throwing something on the floor': 126, 'Tidying something on the floor': 127, 'Holding some medicine': 128, 'Taking/consuming some medicine': 129, 'Putting groceries somewhere': 130, 'Laughing at television': 131, 'Watching television': 132, 'Someone is awakening in bed': 133, 'Lying on a bed': 134, 'Sitting in a bed': 135, 'Fixing a vacuum': 136, 'Holding a vacuum': 137, 'Taking a vacuum from somewhere': 138, 'Washing their hands': 139, 'Fixing a doorknob': 140, 'Grasping onto a doorknob': 141, 'Closing a refrigerator': 142, 'Opening a refrigerator': 143, 'Fixing their hair': 144, 'Working on paper/notebook': 145, 'Someone is awakening somewhere': 146, 'Someone is cooking something': 147, 'Someone is dressing': 148, 'Someone is laughing': 149, 'Someone is running somewhere': 150, 'Someone is going from standing to sitting': 151, 'Someone is smiling': 152, 'Someone is sneezing': 153, 'Someone is standing up from somewhere': 154, 'Someone is undressing': 155, 'Someone is eating something': 156}
        hockey_dict = {'Celebration': 0, 'Checking': 1, 'Corner Action': 2, 'End of Period': 3, 'Face-Off': 4, 'Fight': 5, 'Goal': 6, 'Line Change': 7, 'Penalty': 8, 'Shot': 9, 'Save': 10, 'Play': 11}
        thumos14_dict = {'BaseballPitch': 0, 'BasketballDunk': 1, 'Billiards': 2, 'CleanAndJerk': 3, 'CliffDiving': 4, 'CricketBowling': 5, 'CricketShot': 6, 'Diving': 7, 'FrisbeeCatch': 8, 'GolfSwing': 9, 'HammerThrow': 10, 'HighJump': 11, 'JavelinThrow': 12, 'LongJump': 13, 'PoleVault': 14, 'Shotput': 15, 'SoccerPenalty': 16, 'TennisSwing': 17, 'ThrowDiscus': 18, 'VolleyballSpiking': 19}
        volleyball_dict = {'Blocking': 0, 'Digging': 1, 'Falling': 2, 'Jumping': 3, 'Moving': 4, 'Setting': 5, 'Spiking': 6, 'Standing': 7, 'Waiting': 8}
        if(self.dataset == "animalkingdom"): return animalkingdom_dict
        elif(self.dataset == "charades"): return charades_dict
        elif(self.dataset == "hockey"): return hockey_dict
        elif(self.dataset == "thumos14"): return thumos14_dict
        elif(self.dataset == "volleyball"): return volleyball_dict        

    def get_train_transforms(self,):
        """Returns the training torchvision transformations for each dataset/method.
           If a new method or dataset is added, this file should by modified
           accordingly.
        Args:
          method: The name of the method.
        Returns:
          train_transform: An object of type torchvision.transforms.
        """
        self._check()
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        input_size = 224

        if self.dataset == 'ava':
            # video_transforms = create_video_transform(mode='train',
            #                                           video_key='video',
            #                                           remove_key=['video_name', 'video_index', 'clip_index', 'aug_index', 'boxes', 'extra_info'],
            #                                           num_samples=8,
            #                                           convert_to_float=False)
            # labels_transforms = ApplyTransformToKey(key='labels',
            #     transform=Lambda(lambda nest_list: np.array([(1) if i in set([item-1 for _list in nest_list for item in _list]) else (0) for i in range(80)])))
            # transforms = Compose([video_transforms, labels_transforms])
            raise NotImplementedError
        else:
            unique = Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                              GroupRandomHorizontalFlip(True),
                              GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                              GroupRandomGrayscale(p=0.2),
                              GroupGaussianBlur(p=0.0),
                              GroupSolarization(p=0.0)])
            common = Compose([Stack(roll=False),
                            ToTorchFormatTensor(div=True),
                            GroupNormalize(input_mean, input_std)])
            transforms = Compose([unique, common])
        return transforms
    
    def get_test_transforms(self,):
        """Returns the evaluation torchvision transformations for each dataset/method.
           If a new method or dataset is added, this file should by modified
           accordingly.
        Args:
          method: The name of the method.
        Returns:
          test_transform: An object of type torchvision.transforms.
        """
        self._check()
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        input_size = 224
        scale_size = 256

        if self.dataset == 'ava':
            # video_transforms = create_video_transform(mode='val',
            #                                           video_key='video',
            #                                           remove_key=['video_name', 'video_index', 'clip_index', 'aug_index', 'boxes', 'extra_info'],
            #                                           num_samples=8,
            #                                           convert_to_float=False)
            # labels_transforms = ApplyTransformToKey(key='labels',
            #     transform=Lambda(lambda nest_list: np.array([(1) if i in set([item-1 for _list in nest_list for item in _list]) else (0) for i in range(80)])))
            # transforms = Compose([video_transforms, labels_transforms])
            raise NotImplementedError
        else:
            unique = Compose([GroupScale(scale_size),
                              GroupCenterCrop(input_size)])
            common = Compose([Stack(roll=False),
                              ToTorchFormatTensor(div=True),
                              GroupNormalize(input_mean, input_std)])
            transforms = Compose([unique, common])
        return transforms

    def get_train_loader(self, train_transform, drop_last=False):
        """Returns the training loader for each dataset.
           If a new method or dataset is added, this method should by modified
           accordingly.
        Args:
          path: disk location of the dataset.
          dataset: the name of the dataset.
          total_length: the number of frames in a video clip
          batch_size: the mini-batch size.
          train_transform: the transformations used by the sampler, they
            should be returned by the method get_train_transforms().
          num_workers: the total number of parallel workers for the samples.
          drop_last: it drops the last sample if the mini-batch cannot be
             aggregated, necessary for methods like DeepInfomax.            
        Returns:
          train_loader: The loader that can be used a training time.
        """
        self._check()
        act_dict = self.get_act_dict()
        if(self.dataset == 'animalkingdom'):
            train_data = AnimalKingdom(self.path, act_dict, total_length=self.total_length, transform=train_transform, mode='train')
            sampler = RandomSampler(train_data, num_samples=2500)
            if self.distributed:
                sampler = DistributedSamplerWrapper(sampler, shuffle=True)
            shuffle = False
        elif(self.dataset == 'charades'):
            train_data = Charades(self.path, act_dict, total_length=self.total_length, transform=train_transform, random_shift=False, mode='train')
            sampler = RandomSampler(train_data, num_samples=2500)
            if self.distributed:
                sampler = DistributedSamplerWrapper(sampler, shuffle=True)
            shuffle = False
        elif(self.dataset == 'hockey'):
            train_data = Hockey(self.path, act_dict, total_length=self.total_length, transform=train_transform, mode='train', test_part=self.test_part, stride=10)
            sampler = RandomSampler(train_data, num_samples=2500)
            if self.distributed:
                sampler = DistributedSamplerWrapper(sampler, shuffle=True)
            shuffle = False
        elif(self.dataset == "thumos14"):
            train_data = Thumos14(self.path, act_dict, total_length=self.total_length, transform=train_transform, mode='train')
            sampler = RandomSampler(train_data, num_samples=2500)
            if self.distributed:
                sampler = DistributedSamplerWrapper(sampler, shuffle=True)
            shuffle = False
        elif(self.dataset == 'volleyball'):
            train_data = Volleyball(self.path, act_dict, total_length=self.total_length, transform=train_transform, mode='train')
            sampler = DistributedSampler(train_data, shuffle=True) if self.distributed else None
            shuffle = False
        else:
            raise Exception("[ERROR] The dataset " + str(self.dataset) + " is not supported!")
        
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, sampler=sampler, pin_memory=False, drop_last=drop_last)
        return train_loader
    
    def get_test_loader(self, test_transform, drop_last=False):
        """Returns the test loader for each dataset.
           If a new method or dataset is added, this method should by modified
           accordingly.
        Args:
          path: disk location of the dataset.
          dataset: the name of the dataset.
          total_length: the number of frames in a video clip
          batch_size: the mini-batch size.
          train_transform: the transformations used by the sampler, they
            should be returned by the method get_train_transforms().
          num_workers: the total number of parallel workers for the samples.
          drop_last: it drops the last sample if the mini-batch cannot be
             aggregated, necessary for methods like DeepInfomax.            
        Returns:
          train_loader: The loader that can be used a training time.
        """
        self._check()
        act_dict = self.get_act_dict()
        if(self.dataset == 'animalkingdom'):
            test_data = AnimalKingdom(self.path, act_dict, total_length=self.total_length, transform=test_transform, mode='val')
        elif(self.dataset == 'charades'):
            test_data = Charades(self.path, act_dict, total_length=self.total_length, transform=test_transform, mode='test')
        elif(self.dataset == 'hockey'):
            test_data = Hockey(self.path, act_dict, total_length=self.total_length, transform=test_transform, mode='test', stride=45)
        elif(self.dataset == "thumos14"):
            test_data = Thumos14(self.path, act_dict, total_length=self.total_length, transform=test_transform, mode='test')
        elif(self.dataset == 'volleyball'):
            test_data = Volleyball(self.path, act_dict, total_length=self.total_length, transform=test_transform, mode='test')
        else:
            raise Exception("[ERROR] The dataset " + str(self.dataset) + " is not supported!")
        sampler = DistributedSampler(test_data, shuffle=False) if self.distributed else None
        shuffle = False
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, sampler=sampler, pin_memory=True, drop_last=drop_last)
        return test_loader
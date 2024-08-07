import torch
import os
import PIL
from PIL import ImageEnhance
from .corruptions import *
from .vision import VisionDataset
from .utils import download_file_from_google_drive, check_integrity, Mel


def is_wav_file(file_path):
    # Check if the file exists
    if not os.path.isfile(file_path):
        return False
    
    # Check the file extension
    return file_path.lower().endswith('.wav')

class Audio_Image(VisionDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test'}.
            Accordingly dataset is selected.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    base_folder = ""

    def __init__(self, root = "/home/bvandelft/rcp/scratch/datasets/maestro-v3.0.0/maestro_full",
                 cfg = None, 
                 split="train",
                 test_split=0.1,
                 transform=None, target_transform=None,):
        
        super(Audio_Image, self).__init__(root=root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.list_of_wav_files = []
        if cfg is None:
            self.mel = Mel( x_res = 256,
                            y_res = 256,
                            sample_rate = 22050,
                            n_fft = 2048,
                            hop_length = 512,
                            top_db = 80,
                            n_iter = 32,
                            number_of_slices = -1,)
        else:
            self.mel = Mel( x_res = cfg.trainer.img_size,
                            y_res = cfg.trainer.img_size,
                            sample_rate = cfg.trainer.sample_rate,
                            n_fft = cfg.trainer.n_fft,
                            hop_length = cfg.trainer.hop_length,
                            top_db = cfg.trainer.top_db,
                            n_iter = cfg.trainer.n_iter,
                            number_of_slices = cfg.trainer.number_of_slices,)

        self.all_length = len([name for name in os.listdir(self.root) if is_wav_file(os.path.join(self.root, name))])
        if self.split == "all" :
            self.length = self.all_length
        elif self.split == 'train':
            self.length = int((1-test_split)*self.all_length)
        elif self.split == 'test':
            self.length = int(test_split*self.all_length)
        else:
            print("Assuming to take the whole dataset")
            self.split = "all"
        print("nb of song:", self.length)
        for k, name in enumerate([name for name in os.listdir(self.root) if is_wav_file(os.path.join(self.root, name))]):
            if self.split == 'train' and k < self.length:
                self.list_of_wav_files.append(name)
            elif self.split == "test" and k >= self.all_length - self.length:
                self.list_of_wav_files.append(name)
            elif self.split == 'all':
                self.list_of_wav_files.append(name)
        assert self.length == len(self.list_of_wav_files)


    def __getitem__(self, index):
        wav_path = f"{self.root}/{self.list_of_wav_files[index]}"
        img_slice = self.mel.encode([wav_path])
        # print("Shape of output", img_slice.shape)
        # X = PIL.Image.open(img_path)
        # X_original = X

        if self.target_transform is not None:
            target = self.target_transform(img_slice)
        if self.transform is not None:
            X = self.transform(img_slice)
            X_original = self.transform(img_slice)
        else:
            X = img_slice
            X_original = img_slice
            


        indexes = torch.tensor(index).int()
        return X, X_original, indexes

    def __len__(self):

        return self.length


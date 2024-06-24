import torch
import os
import PIL
from PIL import ImageEnhance
from .corruptions import *
from .vision import VisionDataset
from .utils import download_file_from_google_drive, check_integrity


class Faces(VisionDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                ``identity`` (int): label for each person (data points with the same identity are the same person)
                ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                    righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)
            Defaults to ``attr``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "CelebAMask-HQ"


    def __init__(self, celeba_root = "/home/bvandelft/scitas/bastien/CelebAMask-HQ/CelebA-HQ-img",
                 ffhq_root="/home/bvandelft/scitas/datasets/images1024x1024/",
                 split="train",
                 transform=None, target_transform=None,
                 corruption=None,
                 corruption_severity=5):

        super(Faces, self).__init__(root="/home/bvandelft/scitas/bastien/")
        self.split = split
        self.corruption = corruption
        self.corruption_severity = corruption_severity
        self.transform = transform
        self.target_transform = target_transform
        self.celeba_root = celeba_root
        self.celeba_length = len([name for name in os.listdir(self.celeba_root) if os.path.isfile(os.path.join(self.celeba_root, name))])
        self.ffhq_root = ffhq_root
        self.ffhq_length = len([name for name in os.listdir(self.ffhq_root) if os.path.isfile(os.path.join(self.ffhq_root, name))])


    def __getitem__(self, index):
        if index <= self.celeba_length:
            img_path = f"{self.celeba_root}/{index}.jpg"
            X = PIL.Image.open(img_path)
            X_original = X
        else:
            new_index = str(index - self.celeba_length)
            while len(new_index) < 5:
                new_index = '0'+new_index
            img_path = f"{self.ffhq_root}/{new_index}.png"
            X = PIL.Image.open(img_path)
            X_original = X

        if self.target_transform is not None:
            target = self.target_transform(X_original)
        if self.transform is not None:
            X = self.transform(X)
            X_original = self.transform(X_original)
            


        indexes = torch.tensor(index).int()
        return X, X_original, indexes

    def __len__(self):

        return self.celeba_length + self.ffhq_length


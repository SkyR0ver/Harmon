from torch.utils.data import Dataset
from PIL import Image
import os
import json
import random
import torch
import numpy as np
from einops import rearrange
from xtuner.registry import BUILDER
from src.datasets.utils import crop2square
from glob import glob


class Text2ImageDataset(Dataset):
    def __init__(self,
                 data_path,
                 local_folder,
                 image_size,
                 unconditional=0.1,
                 tokenizer=None,
                 prompt_template=None,
                 max_length=1024,
                 crop_image=True,
                 cap_source='caption',
                 ceph_folder=None,
                 ceph_config=None
                 ):
        pass

    def _load_data(self, data_path):
        pass

    def __len__(self):
        return len(self.data_list)

    def _read_image(self, image_file):
        pass

    def _process_text(self, text):
        pass

    def _process_image(self, image):
        pass

    def _retry(self):
        pass

    def __getitem__(self, idx):
        pass


class LargeText2ImageDataset(Text2ImageDataset):
    # self.data_list only contains paths of images and captions

    def __init__(self, cap_folder=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cap_folder = self.local_folder if cap_folder is None else cap_folder
        self.data_list = []

    def _load_data(self, data_path):      # image path and annotation path are saved in a json file
        pass

    def __getitem__(self, idx):
        pass

from torch.utils.data import Dataset
from PIL import Image
import os
import io
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

class BlipO3Dataset(Text2ImageDataset):

    def __init__(self, 
                 data_path="/scratch/2025_05/jixie/BLIP3o-60k/*.tar",
                 cache_dir='/scratch/2025_05/jixie/',
                 *args, **kwargs):
        self.data_path = data_path
        self.cache_dir = cache_dir
        super().__init__(data_path=data_path, *args, **kwargs)

    def _load_data(self, data_path):
        try:
            from datasets import load_dataset
            print(f"Loading dataset from {data_path} with cache_dir {self.cache_dir}")
            data_files = glob(data_path) 
            self.dataset = load_dataset("webdataset", data_files=data_files, cache_dir=self.cache_dir, split="train", num_proc=64)

            print(f"Loaded {len(self.dataset)} samples from {data_path}")
            
            self.data_list = []
            for idx in range(len(self.dataset)):
                self.data_list.append({
                    'idx': idx,
                })
                
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.data_list = []

        print(f"Load {len(self.data_list)} data samples from {data_path}", flush=True)

    def __getitem__(self, idx):
        try:
            data_sample = self.data_list[idx]
            original_idx = data_sample['idx']
            
            sample = self.dataset[original_idx]
            
            image_data = sample['jpg']
            if isinstance(image_data, dict) and 'bytes' in image_data:
                image = Image.open(io.BytesIO(image_data['bytes'])).convert('RGB')
            elif hasattr(image_data, 'convert'):
                image = image_data.convert('RGB')
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
            else:
                try:
                    image = Image.fromarray(np.array(image_data)).convert('RGB')
                except:
                    raise TypeError(f"无法处理的图像类型: {type(image_data)}")
            
            caption = sample['txt']
            
            data = self._process_image(image)
            data.update(self._process_text(caption))
            data.update(type='text2image')
            return data

        except Exception as e:
            print(f"Error when processing index {idx}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return self._retry()
        
        
class MidJourneyDataset(Text2ImageDataset):
    def __init__(self, 
                 data_path="brivangl/midjourney-v6-llava",
                 cache_dir='/scratch/2025_06/jixie/',
                 *args, **kwargs):
        self.data_path = data_path
        self.cache_dir = cache_dir
        super().__init__(data_path=data_path, *args, **kwargs)

    def _load_data(self, data_path):
        try:
            from datasets import load_dataset
            print(f"Loading dataset from {data_path} with cache_dir {self.cache_dir}")
            self.dataset = load_dataset(data_path, cache_dir=self.cache_dir)['train']
            print(f"Loaded {len(self.dataset)} samples from {data_path}")
            
            self.data_list = []
            for idx in range(len(self.dataset)):
                self.data_list.append({
                    'idx': idx, 
                })
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.data_list = []

        print(f"Load {len(self.data_list)} data samples from {data_path}", flush=True)

    def __getitem__(self, idx):
        try:
            data_sample = self.data_list[idx]
            original_idx = data_sample['idx']
            
            sample = self.dataset[original_idx]
            
            image_data = sample['image']
            if isinstance(image_data, dict) and 'bytes' in image_data:
                image = Image.open(io.BytesIO(image_data['bytes'])).convert('RGB')
            elif hasattr(image_data, 'convert'):
                image = image_data.convert('RGB')
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
            else:
                try:
                    image = Image.fromarray(np.array(image_data)).convert('RGB')
                except:
                    raise TypeError(f"无法处理的图像类型: {type(image_data)}")
            
            caption = sample['prompt']
            
            data = self._process_image(image)
            data.update(self._process_text(caption))
            data.update(type='text2image')
            return data

        except Exception as e:
            print(f"Error when processing index {idx}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return self._retry()
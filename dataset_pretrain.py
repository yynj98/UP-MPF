import cv2
import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torchvision import transforms


class RotationImgDataset(Dataset):
    def __init__(self, args, mode='train'):
        super().__init__()
        self.mode = mode

        self.img_token = args.img_token
        self.img_token_len = args.img_token_len

        self.data_dir = args.data_dir
        self.img_dir = args.img_dir

        self.template = ['[CLS] Image : " ', ' " . Rotation direction of the image : [MASK] . [SEP]']
        self.label_list = ['keep', 'left', 'right']

        self.tokenizer = BertTokenizer.from_pretrained(args.model_name)
        self.vocab = self.tokenizer.get_vocab()

        self.label_id_list = [self.vocab[token] for token in self.label_list]
        self.label_id_map = {key: self.vocab[key] for key in self.label_list}

        if mode == 'train':
            self.lines = self._read_txt(os.path.join(self.data_dir, 'train.txt'))
            self.train_indices = list(range(len(self.lines)))
            self.keep_range = len(self.train_indices) // 3
            self.left_range = len(self.train_indices) // 3 * 2
            random.shuffle(self.train_indices)
        else:
            self.lines = self._read_txt(os.path.join(self.data_dir, 'test.txt'))

    def _read_txt(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            return f.readlines()

    def shuffle(self):
        random.shuffle(self.train_indices)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        if self.mode == 'test':
            img_id, rotation_direction = line.split()
        else:
            img_id = line.rstrip()
            if idx in self.train_indices[:self.keep_range]:
                rotation_direction = 'keep'
            elif idx in self.train_indices[self.keep_range: self.left_range]:
                rotation_direction = 'left'
            else:
                rotation_direction = 'right'
        
        input_tokens = self.tokenizer.tokenize(self.template[0]) + [self.img_token] * self.img_token_len + self.tokenizer.tokenize(self.template[1])
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(input_ids)
        for i, id in enumerate(input_ids):
            if id == self.tokenizer.mask_token_id:
                labels[i] = self.label_id_map[rotation_direction]
        
        return {
            'img_id': img_id,
            'rotation_direction': rotation_direction,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def collate_fn(self, batch):
        input_ids = torch.tensor([instance["input_ids"] for instance in batch])
        attention_mask = torch.tensor([instance["attention_mask"] for instance in batch])
        labels = torch.tensor([instance["labels"] for instance in batch])

        imgs = []
        img_ids = [instance["img_id"] for instance in batch]
        rot_dirs = [instance["rotation_direction"] for instance in batch]
        for img_id, rot_dir in zip(img_ids, rot_dirs):
            img = cv2.imread(os.path.join(self.img_dir, img_id))
            img = img[:,:,::-1]  # BGR -> RGB
            if rot_dir == 'left':
                img = np.rot90(img, 1)
            elif rot_dir == 'right':
                img = np.rot90(img, -1)
            else:
                pass
            img = np.ascontiguousarray(img)
            img = transforms.ToTensor()(img)
            
            if self.mode == 'train':
                img = transforms.Resize([256, 256])(img)
                img = transforms.RandomCrop([224, 224])(img)
            else:
                img = transforms.Resize([224, 224])(img)
            
            imgs.append(img)
        imgs = torch.stack(imgs, dim=0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            'imgs': imgs,
        }

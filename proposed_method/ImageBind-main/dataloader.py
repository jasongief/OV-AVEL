#! for baseline_v0_training_free.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from config import cfg
import imagebind
# from imagebind import data

import os
import os.path as osp
import numpy as np
import pandas as pd
import csv
import json
import pdb


class OVAVE_Dataset(Dataset):
    def __init__(self, split, test_data_type='total', device=None, debug=False):
        meta_csv_path = cfg.META_CSV_PATH
        meta_df = pd.read_csv(meta_csv_path)
        anno_json_path = cfg.ANNO_JSON_PATH
        with open(anno_json_path, 'r') as fr:
            self.anno_data = json.load(fr)
        self.data_base_dir = cfg.DATA_BASE_DIR
        self.split = split
        self.test_data_type = test_data_type
        self.split_meta_df = meta_df[meta_df['split'] == self.split]
        if self.split in ['test', 'val']:
            if self.test_data_type != 'total':
                self.split_meta_df = self.split_meta_df[ self.split_meta_df['cls_type'] == test_data_type ]

        self.debug = debug
        if self.debug:
            self.split_meta_df = self.split_meta_df.tail(64)
        # self.device = device
        # if self.device is None:
        #     # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        #     self.device = "cpu"

        if self.split == 'train':
            self.text_list = pd.read_csv(cfg.TRAIN_CLOSE_CATEGORY_CSV_PATH, header=None)[0].tolist()
        else:
            self.text_list = pd.read_csv(cfg.TOTAL_CLOSE_OPEN_CATEGORY_CSV_PATH, header=None)[0].tolist()
        self.text_inputs   = imagebind.data.load_and_transform_text(self.text_list) #! [bs, 46, 77] 
 

    def __len__(self):
        return len(self.split_meta_df)


    def _gene_event_labels_train(self, category, vid_name):
        avc_label = json.loads(self.anno_data[vid_name]['label'])
        avc_label = torch.FloatTensor(avc_label)
        if torch.sum(avc_label) == 0:
            category_label = torch.LongTensor([cfg.TRAIN_BG_CLASS_ID]) #! background video
        else:
            train_close_categories = pd.read_csv(cfg.TRAIN_CLOSE_CATEGORY_CSV_PATH, header=None)[0].tolist()
            # label_mat = torch.zeros([10, len(train_close_categories)]) # [10, 46]
            label_idx = train_close_categories.index(category)
            # label_mat[:, label_idx] = avc_label
            # return label_mat.to(self.device)
            # category_label = torch.zeros([len(train_close_categories)])
            # category_label[label_idx] = 1
            category_label = torch.LongTensor([label_idx])
        return avc_label, category_label

    def _gene_event_labels_test(self, category, vid_name):
        avc_label = json.loads(self.anno_data[vid_name]['label'])
        avc_label = torch.FloatTensor(avc_label)
        if torch.sum(avc_label) == 0:
            category_label = torch.LongTensor([cfg.TOTAL_BG_CLASS_ID]) #! background video
        else:
            test_close_open_categories = pd.read_csv(cfg.TOTAL_CLOSE_OPEN_CATEGORY_CSV_PATH, header=None)[0].tolist()
            # label_mat = torch.zeros([10, len(test_close_open_categories)]) # [10, 67]
            label_idx = test_close_open_categories.index(category)
            # label_mat[:, label_idx] = avc_label
            # return label_mat.to(self.device)
            # category_label = torch.zeros([len(test_close_open_categories)])
            # category_label[label_idx] = 1
            category_label = torch.LongTensor([label_idx])
        return avc_label, category_label


    def __getitem__(self, idx):
        _, category, cls_type, vid_name = self.split_meta_df.iloc[idx, :]
        audio_paths = [osp.join(self.data_base_dir, self.split, 'audio', category, vid_name+'.wav')]
        visual_frames_dir = osp.join(self.data_base_dir, self.split, 'video', category, vid_name)
        visual_frames_paths = []
        assert len(os.listdir(visual_frames_dir)) == 10
        img_names = os.listdir(visual_frames_dir)
        img_names.sort()
        for img_name in img_names:
            visual_frames_paths.append(osp.join(visual_frames_dir, img_name))
        
        audio_inputs  = imagebind.data.load_and_transform_audio_data(audio_paths) # [bs, 1, 10, 1, 128, 204]
        visual_inputs = imagebind.data.load_and_transform_vision_data(visual_frames_paths) # [bs, 10, 3, 224, 224]

        if self.split == 'train':
            avc_label, category_label = self._gene_event_labels_train(category, vid_name) # [bs, 10, 46]
        else:
            avc_label, category_label = self._gene_event_labels_test(category, vid_name) # [bs, 10, 67]

        return audio_inputs, visual_inputs, self.text_inputs, avc_label, category_label, vid_name
        # return audio_inputs, visual_inputs, avc_label, category_label, vid_name



if __name__ == "__main__":
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType
    from einops import rearrange, repeat, reduce

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)


    # split = 'train'
    split = 'test'
    BS = 16
    split_dataset = OVAVE_Dataset(split, device=None, debug=True)
    split_dataloader = torch.utils.data.DataLoader(
        split_dataset,
        batch_size=BS,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    ) 
    for n_iter, batch_data in enumerate(split_dataloader):
        audio, visual, text, avc_label, category_label, vid_name = batch_data
        # text = split_dataset.text_inputs
        avc_label = avc_label.cuda()
        category_label = category_label.cuda()
        pdb.set_trace()

        audio = audio.squeeze(1) # [bs, 10, 1, 128, 204]]
        print(vid_name)
        pdb.set_trace()
        # text = rearrange(text, 'b k d -> (b k) d')
        single_text = text[0]
        inputs = {
            ModalityType.TEXT: single_text.cuda(), # [46/67, 77]
            ModalityType.VISION: visual.cuda(), # [bs, 10, 3, 224, 224]
            ModalityType.AUDIO: audio.cuda(), # [bs, 10, 1, 128, 204]
        }
        with torch.no_grad():
            embeddings = model(inputs)
            # audio: [bs, 10, 1024], vision: [bs, 10, 1024], text: [46 or 67, 1024]
    pdb.set_trace()
import jieba
import jieba.posseg as pseg
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import json
import pickle 
import torch
from torch.nn.utils.rnn import pad_sequence

def generate_pos_tagging(texts, tag_json = '../data/pos_tags.json', same_size = True, tqdm_mode = True, n_jobs = 6):
    """generate pos tagging array

    Args:
        texts (array): array of raw sentence data
        tag_json (str, optional): pos_tags dict file. Defaults to 'pos_tags.json'.
        same_size (bool, optional): output same size as the original data. Defaults to True.
        tqdm_mode (bool, optional): visualize with tqdm. Defaults to True.
        n_jobs (int, optional): parallel jobs number. Defaults to 6.

    Returns:
        array(int): array of corresponding pos tag numbers to each sentence
    """
    remove_space_txt = list(map(_remove_space, texts))
    
    with open(tag_json, 'r') as f:
        tag_dict= json.load(f)
    
    iterate = tqdm(remove_space_txt) if tqdm_mode else remove_space_txt
    pos_tags = Parallel(n_jobs=n_jobs)(delayed(
        _pos_tagger)(txt,same_size,tag_dict) for txt in iterate)

    return pos_tags
        
def _pos_tagger(text, same_size,tag_dict):
    pos_tags = []
    jieba.enable_paddle()

    if same_size:
        tags = []
        for word in pseg.cut(text, use_paddle=True):
            pos_tags.extend([tag_dict[word.flag.lower()]]*len(word.word))

    else:
        pos_tags = [word.flag for word in pseg.cut(text, use_paddle=True)]
    return pos_tags


def _remove_space(x):
    return x.replace(" ","")

    
def load_pos_tag_tensor_emb(tag_data,tag_json = 'pos_tags.json'):
    """transform tag data into torch tensor

    Args:
        tag_data (array or str): tag data or path to the tag data pickle file
        tag_json (str, optional): path to the pos_tags json file. Defaults to 'pos_tags.json'.

    Returns:
        torch.tensor: tensor of the pos tag data
    """
    with open(tag_json, 'r') as f:
        tag_dict= json.load(f)
    
    if isinstance(tag_data, str):
        with open(tag_data, 'rb') as f:
            tag_data = pickle.load(f)
    
    feature_arr = [torch.tensor(data) for data in tag_data]
    feature_tensor = torch.transpose(pad_sequence(feature_arr, padding_value=tag_dict['PAD']),0,1)
    
    # add padding
    temp = torch.zeros((feature_tensor.size(0),1))
    feature_tensor = torch.cat((feature_tensor, temp), 1)
    feature_tensor = torch.cat((temp, feature_tensor), 1)

    return feature_tensor
__author__ = "Shihao Lin"
__time__   = "2021/11/22"
__version__= "1.0"

import os

from torch import nn


def generate_Pinyin_initial_weight(pinyin_filepath,embedding_size =30):
    """
    filepath: path to char_map
    return a Glyph Embedding layer weights
        line 0 is reserved for special_tokens like [CLS] [PAD] [SEP]
        line 1 is reserved for [UNK]
    """

    with open(pinyin_filepath,'r',encoding='utf-8') as f:
        pinyin2idx = json.load(f)
        matrix = np.zeros((len(pinyin2idx)+2,embedding_size),dtype='float32')
        matrix[1] = np.random.uniform(-0.1,0.1,embedding_size) # unknown words
        
        for i, char in enumerate(.keys(),2):
            # shape: font_size,font_size,3
            try:

                char_array = array3d(font.render(char,True,(0,0,0),(255,255,255)))
                # Convert rgb [0,0,0] to 1.
                char_array = 1-char_array[:font_size,:font_size,0].reshape(-1)/255
                matrix[i][:char_array.shape[0]] = char_array
            except:
                print(char)
    return matrix
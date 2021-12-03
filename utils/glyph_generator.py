__author__ = "Shihao Lin"
__time__   = "2021/11/22"
__version__= "1.0"

import json
import numpy as np
import pygame
from pygame.surfarray import array3d

"""
This script is used to produce the pretrained Glyph embedding data
"""

def generate_Glyph_initial_weight(chardict_filepath,font_filepath,font_size=24):
    """
    filepath: path to char_map
    return a Glyph Embedding layer weights
        line 0 is reserved for special_tokens like [CLS] [PAD] [SEP]
        line 1 is reserved for [UNK]
    """

    with open(chardict_filepath,'r',encoding='utf-8') as f:
        
        pygame.init()
        font = pygame.font.Font(font_filepath,font_size) #仿宋
        
        char2idx = json.load(f)
        matrix = np.zeros((len(char2idx)+2,font_size**2),dtype='float32')
        matrix[1] = np.random.uniform(-0.1,0.1,font_size**2) # unknown words
        
        for i, char in enumerate(char2idx.keys(),2):
            # shape: font_size,font_size,3
            try:

                char_array = array3d(font.render(char,True,(0,0,0),(255,255,255)))
                # Convert rgb [0,0,0] to 1.
                char_array = 1-char_array[:font_size,:font_size,0].reshape(-1)/255
                matrix[i][:char_array.shape[0]] = char_array
            except:
                print(char)
    return matrix

if __name__ == '__main__':
    print('Start to generate ...')
    matrix = generate_Glyph_initial_weight('../data/char_map.json','simfang.ttf')
    np.save('../data/glyph_weight',matrix)
    print('Generation complete ...')
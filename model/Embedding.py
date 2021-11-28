__author__ = "Shihao Lin"
__time__   = "2021/11/28"
__version__= "1.0"

import numpy as np
import torch,json
from torch import nn

class GlyphEmbedding(nn.Module):
    
    def __init__(self, font_weight_path:str):
        super().__init__()
        font_weights = np.load(font_weight_path).astype(np.float32)
        
        self.char_dim = font_weights.shape[0] # numbers of unique char
        self.embbeding_dim = font_weights.shape[1]
        
        self.embedding = nn.Embedding(
            num_embeddings=self.char_dim,
            embedding_dim=self.embbeding_dim,
            _weight=torch.from_numpy(font_weights)
        )
    def forward(self,input_ids):
        return self.embedding(input_ids)
    
class PinyinEmbedding(nn.Module):
    
    def __init__(self, embedding_dim:int,pinyin_path:str):
        super().__init__()
        with open(pinyin_path,'r',encoding='utf-8') as f:
            pinyin2idx = json.load(f)
        
        self.embedding = nn.Embedding(
            num_embeddings=len(pinyin2idx)+2, # 0 for padding, 1 for unknown 
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        # initialize the embedding weights through uniformly sample from [-square_root(3/dim), +square_root(3/dim)]
        self.embedding.weight.data.uniform_(-(3/embedding_dim)**0.5, (3/embedding_dim)**0.5)
    def forward(self,input_ids):
        return self.embedding(input_ids) # [batch, sentence length , embedding dimension]
    
class FusionEmbedding(nn.Module):
    """
    Word Embedding + Char Embedding + Glyph Embedding + Position Embedding
    """
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.position_embeddings = nn.Embedding(config['max_position_embeddings'],config['hidden_size'])
        self.glyph_embeddings = GlyphEmbedding(config['font_weight_path'])
        self.pinyin_embeddings = PinyinEmbedding(config['pinyin_embed_dim'],config['pinyin_path'])
        
        self.pos_tag_embeddings = nn.Embedding(config['tag_size'], config['tag_emb_dim'], padding_idx = 0)
        # initialize the embedding weights through uniformly sample from [-square_root(3/dim), +square_root(3/dim)]
        self.pos_tag_embeddings.weight.data.uniform_(-(3/config['tag_emb_dim'])**0.5, (3/config['tag_emb_dim'])**0.5)
        
        self.fc = nn.Linear(config['hidden_size']+config['pinyin_embed_dim'] \
                            +24**2 + config['tag_emb_dim'], config['hidden_size'])
        
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.dropout = nn.Dropout(config['hidden_dropout'])

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config['max_position_embeddings']).expand((1, -1)))
        
    def forward(self,word_embeddings,pinyin_ids,glyph_ids,pos_ids):
        batch_size = pinyin_ids.shape[0]
        seq_length = pinyin_ids.shape[1]
        position_ids = self.position_ids[:,:seq_length]
        
        pinyin_embeddings = self.pinyin_embeddings(pinyin_ids)
        glyph_embeddings = self.glyph_embeddings(glyph_ids)
        pos_tag_embeddings = self.pos_tag_embeddings(pos_ids)
        
        concat_embeddings = torch.cat((word_embeddings,pinyin_embeddings,glyph_embeddings,pos_tag_embeddings),2)
        
        fusion_embed = self.fc(concat_embeddings)
        
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = fusion_embed + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        return self.dropout(embeddings)
    

class BertEmbedding(nn.Module):
    """
    Bert Word Embedding + Position Embedding
    """
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.position_embeddings = nn.Embedding(config['max_position_embeddings'],config['hidden_size'])
        
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.dropout = nn.Dropout(config['hidden_dropout'])

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config['max_position_embeddings']).expand((1, -1)))
        
    def forward(self,word_embeddings):
        seq_length = word_embeddings.shape[1]

        position_ids = self.position_ids[:,:seq_length]
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = word_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        return self.dropout(embeddings)
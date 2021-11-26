__author__ = "Shihao Lin"
__time__   = "2021/11/26"
__version__= "1.0"

import torch
from torch import nn
import torch.nn.functional as F
from .Embedding import FusionEmbedding



class Fusion_Anchi_Trans_Decoder(nn.Module):
    """
    Fusion Layer (Pinyin + Glyph + Pos + Anchi Bert last hidden layer) as Multi-Head Layer's memory
    and Output Embedding to Transformer Decoder
    
    # https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html
    
    # config = {
    #     'max_position_embeddings':50,
    #     'hidden_size':768,
    #     'font_weight_path':'data/glyph_weight.npy',
    #     'pinyin_embed_dim':30,
    #     'pinyin_path':'data/pinyin_map.json',
    #     'tag_size':30,
    #     'tag_emb_dim':10,
    #     'layer_norm_eps':1e-12,
    #     'hidden_dropout':0.1,
    #     'nhead':12,
    #     'num_layers':6,
    #     'output_dim':21128
    # }
    """
    def __init__(self,config):
        super().__init__()
        
        self.embedding = FusionEmbedding(config)

        decoder_layer = nn.TransformerDecoderLayer(d_model=config['hidden_size'], nhead=config['nhead'])
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config['num_layers'])
        
        self.Linear = nn.Linear(config['hidden_size'],config['output_dim'])
    def forward(self,Xword_embeddings,Xsents_pinyin_ids, \
                Xsents_token_type_ids,Xsents_pos_ids,\
                Yword_embeddings,Ysents_pinyin_ids, \
                Ysents_token_type_ids,Ysents_pos_ids,
                Xpad_hidden_mask,Ypad_hidden_mask):
        
        memory = self.embedding(Xword_embeddings,Xsents_pinyin_ids, \
                                Xsents_token_type_ids,Xsents_pos_ids).permute([1,0,2])
        tgt = self.embedding(Yword_embeddings,Ysents_pinyin_ids, \
                             Ysents_token_type_ids,Ysents_pos_ids).permute([1,0,2])
        outputs = self.transformer_decoder(tgt, memory,
                                           memory_key_padding_mask =Xpad_hidden_mask, # Xpad_hidden_mask == Xsents_attention_mask.bool()
                                           tgt_key_padding_mask=Ypad_hidden_mask  # Ypad_hidden_mask == Ysents_attention_mask.bool()
                                          )
        outputs = self.Linear(outputs).permute([1,0,2])
        
        return F.log_softmax(outputs, dim=1) #[batch_size,max_length,output_dim]
    
class Fusion_Anchi_Transformer(nn.Module):
    """
    Transformer with
    Fusion Layer (Pinyin + Glyph + Pos + Anchi Bert last hidden layer)
    as Input Embedding and Output Embedding
    
    # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
    
    d_model: int = 512,
    nhead: int = 8,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    activation: str = 'relu',
    custom_encoder: Union[Any, NoneType] = None,
    custom_decoder: Union[Any, NoneType] = None,
    """
    
    def __init__(self,config):
        super().__init__()
        
        self.embedding = FusionEmbedding(config)
        self.transformer = nn.Transformer(d_model=config['hidden_size'],nhead=config['nhead'],\
                                          num_encoder_layers=config['num_encoder_layers'],\
                                          num_decoder_layers=config['num_decoder_layers'],\
                                          dim_feedforward=config['dim_feedforward'],\
                                          dropout = config['trans_dropout'],\
                                          activation= config['activation'],\
                                         )
        self.Linear = nn.Linear(config['hidden_size'],config['output_dim'])

    def forward(self,Xword_embeddings,Xsents_pinyin_ids, \
                Xsents_token_type_ids,Xsents_pos_ids,\
                Yword_embeddings,Ysents_pinyin_ids, \
                Ysents_token_type_ids,Ysents_pos_ids,
                Xpad_hidden_mask,Ypad_hidden_mask):
        scr = self.embedding(Xword_embeddings,Xsents_pinyin_ids, \
                                Xsents_token_type_ids,Xsents_pos_ids).permute([1,0,2])
        tgt = self.embedding(Yword_embeddings,Ysents_pinyin_ids, \
                             Ysents_token_type_ids,Ysents_pos_ids).permute([1,0,2])
        
        outputs = self.transformer(scr, tgt,
                                   src_key_padding_mask =Xpad_hidden_mask, # Xpad_hidden_mask == Xsents_attention_mask.bool()
                                   tgt_key_padding_mask=Ypad_hidden_mask  # Ypad_hidden_mask == Ysents_attention_mask.bool()                      
                                  )
        outputs = self.Linear(outputs).permute([1,0,2])
        
        return F.log_softmax(outputs, dim=1) #[batch_size,max_length,output_dim]
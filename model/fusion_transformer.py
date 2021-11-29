__author__ = "Shihao Lin"
__time__   = "2021/11/28"
__version__= "1.0"

import torch
from torch import nn
import torch.nn.functional as F
from .Embedding import FusionEmbedding,BertEmbedding


class Fusion_Anchi_Trans_Decoder(nn.Module):
    """
    Fusion Layer (Pinyin + Glyph + Pos + Anchi Bert last hidden layer)
    + Postional Embedding
    as Multi-Head Layer's memory
    and Output Embedding to Transformer Decoder
    
    # https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html
    
    config = { # for Fusion_Anchi_Trans_Decoder
        'max_position_embeddings':50,
        'hidden_size':768,
        'font_weight_path':'data/glyph_weight.npy',
        'pinyin_embed_dim':30,
        'pinyin_path':'data/pinyin_map.json',
        'tag_size':30,
        'tag_emb_dim':10,
        'layer_norm_eps':1e-12,
        'hidden_dropout':0.1,
        'nhead':12,
        'num_layers':6,
        'output_dim':21128 # fixed
    }
    """
    def __init__(self,config):
        super().__init__()
        
        self.device = config['device']
        self.embedding = FusionEmbedding(config)

        decoder_layer = nn.TransformerDecoderLayer(d_model=config['hidden_size'], nhead=config['nhead'])
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config['num_layers'])
        
        self.Linear = nn.Linear(config['hidden_size'],config['output_dim'])
        self.Linear.weight.data.uniform_(-(3/config['output_dim'])**0.5,\
                                         (3/config['output_dim'])**0.5,)
        self.Linear.bias.data.fill_(0)
        
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self,Xword_embeddings,Xsents_pinyin_ids, \
                Xsents_glyph_ids,Xsents_pos_ids,\
                Yword_embeddings,Ysents_pinyin_ids, \
                Ysents_glyph_ids,Ysents_pos_ids,\
                Xpad_hidden_mask,Ypad_hidden_mask,\
                tgt_mask=None,*args,**kwargs):
        
        memory = self.embedding(Xword_embeddings,Xsents_pinyin_ids, \
                                Xsents_glyph_ids,Xsents_pos_ids).permute([1,0,2])
        tgt = self.embedding(Yword_embeddings,Ysents_pinyin_ids, \
                             Ysents_glyph_ids,Ysents_pos_ids).permute([1,0,2])
        
        if tgt_mask==None:
            tgt_mask = self._generate_square_subsequent_mask(Ypad_hidden_mask.shape[1]).to(self.device)
        
        outputs = self.transformer_decoder(tgt, memory, \
                                           tgt_mask= tgt_mask, \
                                           # Xpad_hidden_mask == ~ Xsents_attention_mask.bool()
                                           memory_key_padding_mask = Xpad_hidden_mask, \
                                           # Ypad_hidden_mask == ~ Ysents_attention_mask.bool()
                                           tgt_key_padding_mask= Ypad_hidden_mask  \
                                          ).permute([1,0,2]) #[batch_size,max_length,output_dim]
        # print('shape1',outputs.shape)

        outputs = self.Linear(outputs)

        return F.log_softmax(outputs, dim=-1)
    
    def encode(self,Xword_embeddings,Xsents_pinyin_ids, \
                Xsents_glyph_ids,Xsents_pos_ids,*args,**kwargs):
        # [max_length, batch_size, hidden_dim]
        return self.embedding(Xword_embeddings,Xsents_pinyin_ids, \
                              Xsents_glyph_ids,Xsents_pos_ids).permute([1,0,2])
    
    
    def decode(self,memory, Xpad_hidden_mask, \
                Yword_embeddings,Ysents_pinyin_ids, \
                Ysents_glyph_ids,Ysents_pos_ids,\
                Ypad_hidden_mask,\
                tgt_mask=None,*args,**kwargs):

        if tgt_mask==None:
            tgt_mask = self._generate_square_subsequenhut_mask(Ypad_hidden_mask.shape[1]).to(self.device)
            
        # [max_length, batch_size, hidden_dim]
        return self.transformer_decoder(tgt=self.embedding(Yword_embeddings,Ysents_pinyin_ids, \
                                                        Ysents_glyph_ids,Ysents_pos_ids).permute([1,0,2]),\
                                        memory = memory,
                                        tgt_mask = tgt_mask,
                                        tgt_key_padding_mask= Ypad_hidden_mask,
                                        memory_key_padding_mask = Xpad_hidden_mask)
    
    

class Fusion_Anchi_Transformer(nn.Module):
    """
    Transformer with
    Fusion Layer (Pinyin + Glyph + Pos + Anchi Bert last hidden layer) + Postional Embedding
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
        self.device = config['device']
        
        self.embedding = FusionEmbedding(config)
        self.transformer = nn.Transformer(d_model=config['hidden_size'],nhead=config['nhead'],\
                                          num_encoder_layers=config['num_encoder_layers'],\
                                          num_decoder_layers=config['num_decoder_layers'],\
                                          dim_feedforward=config['dim_feedforward'],\
                                          dropout = config['trans_dropout'],\
                                          activation= config['activation'],\
                                         )
        self.Linear = nn.Linear(config['hidden_size'],config['output_dim'])
        self.Linear.weight.data.uniform_(-(3/config['output_dim'])**0.5,\
                                         (3/config['output_dim'])**0.5,)
        self.Linear.bias.data.fill_(0)
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self,Xword_embeddings,Xsents_pinyin_ids, \
                Xsents_glyph_ids,Xsents_pos_ids,\
                Yword_embeddings,Ysents_pinyin_ids, \
                Ysents_glyph_ids,Ysents_pos_ids, \
                Xpad_hidden_mask,Ypad_hidden_mask, \
                tgt_mask=None,*args,**kwargs):
        scr = self.embedding(Xword_embeddings,Xsents_pinyin_ids, \
                                Xsents_glyph_ids,Xsents_pos_ids).permute([1,0,2])
        tgt = self.embedding(Yword_embeddings,Ysents_pinyin_ids, \
                             Ysents_glyph_ids,Ysents_pos_ids).permute([1,0,2])
        if tgt_mask==None:
            tgt_mask = self._generate_square_subsequent_mask(Ypad_hidden_mask.shape[1]).to(self.device)
        
        outputs = self.transformer(scr, tgt,\
                                   tgt_mask=tgt_mask,\
                                   # Xpad_hidden_mask == Xsents_attention_mask.bool()
                                   src_key_padding_mask =Xpad_hidden_mask, \
                                   # Ypad_hidden_mask == Ysents_attention_mask.bool()   
                                   tgt_key_padding_mask=Ypad_hidden_mask)
        outputs = self.Linear(outputs).permute([1,0,2])
        
        return F.log_softmax(outputs, dim=-1) #[batch_size,max_length,output_dim]
    
    def encode(self,Xword_embeddings,Xsents_pinyin_ids, \
                Xsents_glyph_ids,Xsents_pos_ids,Xpad_hidden_mask, \
               *args,**kwargs):
        # [max_length, batch_size, hidden_dim]
        return self.transformer.encoder(src=self.embedding(Xword_embeddings,Xsents_pinyin_ids, \
                                        Xsents_glyph_ids,Xsents_pos_ids).permute([1,0,2]),\
                                        src_key_padding_mask=Xpad_hidden_mask)
    
    
    def decode(self,memory, Xpad_hidden_mask, \
                Yword_embeddings,Ysents_pinyin_ids, \
                Ysents_glyph_ids,Ysents_pos_ids,\
                Ypad_hidden_mask,\
                tgt_mask=None,*args,**kwargs):
        
        if tgt_mask==None:
            tgt_mask = self._generate_square_subsequenhut_mask(Ypad_hidden_mask.shape[1]).to(self.device)
            
        # [max_length, batch_size, hidden_dim]
        return self.transformer.decoder(tgt=self.embedding(Yword_embeddings,Ysents_pinyin_ids, \
                                                        Ysents_glyph_ids,Ysents_pos_ids).permute([1,0,2]),\
                                        memory = memory,
                                        tgt_mask = tgt_mask,
                                        tgt_key_padding_mask= Ypad_hidden_mask,
                                        memory_key_padding_mask = Xpad_hidden_mask)
    

    
class Anchi_Transformer(nn.Module):
    """
    Transformer with
    Anchi Bert last hidden layer + Postional Embedding
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
        self.device = config['device']
        self.embedding = BertEmbedding(config)
        self.transformer = nn.Transformer(d_model=config['hidden_size'],nhead=config['nhead'],\
                                          num_encoder_layers=config['num_encoder_layers'],\
                                          num_decoder_layers=config['num_decoder_layers'],\
                                          dim_feedforward=config['dim_feedforward'],\
                                          dropout = config['trans_dropout'],\
                                          activation= config['activation'],\
                                         )
        self.Linear = nn.Linear(config['hidden_size'],config['output_dim'])
        self.Linear.weight.data.uniform_(-(3/config['output_dim'])**0.5,\
                                         (3/config['output_dim'])**0.5,)
        self.Linear.bias.data.fill_(0)
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self,Xword_embeddings, Yword_embeddings, \
                Xpad_hidden_mask,Ypad_hidden_mask,\
                tgt_mask=None,*args,**kwargs):
        
        scr = self.embedding(Xword_embeddings).permute([1,0,2])
        tgt = self.embedding(Yword_embeddings).permute([1,0,2])
        
        if tgt_mask==None:
            tgt_mask = self._generate_square_subsequent_mask(Ypad_hidden_mask.shape[1]).to(self.device)
        
        outputs = self.transformer(scr, tgt,
                                   tgt_mask=tgt_mask,
                                   src_key_padding_mask =Xpad_hidden_mask, # Xpad_hidden_mask == Xsents_attention_mask.bool()
                                   tgt_key_padding_mask=Ypad_hidden_mask  # Ypad_hidden_mask == Ysents_attention_mask.bool()                      
                                  )
        outputs = self.Linear(outputs).permute([1,0,2])
        
        return F.log_softmax(outputs, dim=-1) #[batch_size,max_length,output_dim]
    
    def encode(self,Xword_embeddings, Xpad_hidden_mask,*args,**kwargs):
        # [max_length, batch_size, hidden_dim]
        return self.transformer.encoder(src=self.embedding(Xword_embeddings).permute([1,0,2]),
                                        src_key_padding_mask=Xpad_hidden_mask) 
    
    def decode(self,memory, Xpad_hidden_mask, Yword_embeddings, Ypad_hidden_mask, tgt_mask=None, *args, **kwargs):
        if tgt_mask==None:
            tgt_mask = self._generate_square_subsequenhut_mask(Ypad_hidden_mask.shape[1]).to(self.device)
         # [max_length, batch_size, hidden_dim]
        return self.transformer.decoder(tgt=self.embedding(Yword_embeddings).permute([1,0,2]),
                                        memory = memory,
                                        tgt_mask = tgt_mask,
                                        tgt_key_padding_mask= Ypad_hidden_mask,
                                        memory_key_padding_mask = Xpad_hidden_mask)
    
    
class Anchi_Decoder(nn.Module):
    """
    Anchi Bert last hidden layer
    + Postional Embedding
    as Multi-Head Layer's memory
    and Output Embedding to Transformer Decoder
    
    # https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html
    
    config = { # for Fusion_Anchi_Trans_Decoder
        'max_position_embeddings':50,
        'hidden_size':768,
        'font_weight_path':'data/glyph_weight.npy',
        'pinyin_embed_dim':30,
        'pinyin_path':'data/pinyin_map.json',
        'tag_size':30,
        'tag_emb_dim':10,
        'layer_norm_eps':1e-12,
        'hidden_dropout':0.1,
        'nhead':12,
        'num_layers':6,
        'output_dim':21128 # fixed
    }
    """
    def __init__(self,config):
        super().__init__()
        
        self.device = config['device']
        self.embedding = BertEmbedding(config)

        decoder_layer = nn.TransformerDecoderLayer(d_model=config['hidden_size'], nhead=config['nhead'])
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config['num_layers'])
        
        self.Linear = nn.Linear(config['hidden_size'],config['output_dim'])
        self.Linear.weight.data.uniform_(-(3/config['output_dim'])**0.5,\
                                         (3/config['output_dim'])**0.5,)
        self.Linear.bias.data.fill_(0)
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self,Xword_embeddings,\
                Yword_embeddings,\
                Xpad_hidden_mask,Ypad_hidden_mask,\
                tgt_mask=None,*args,**kwargs):
        
        memory = self.embedding(Xword_embeddings).permute([1,0,2])
        tgt = self.embedding(Yword_embeddings).permute([1,0,2])
        
        
        if tgt_mask==None:
            tgt_mask = self._generate_square_subsequenhut_mask(Ypad_hidden_mask.shape[1]).to(self.device)
        
        outputs = self.transformer_decoder(tgt=tgt,memory=memory, \
                                           tgt_mask= tgt_mask, \
                                           # Xpad_hidden_mask == ~ Xsents_attention_mask.bool()
                                           memory_key_padding_mask = Xpad_hidden_mask, \
                                           # Ypad_hidden_mask == ~ Ysents_attention_mask.bool()
                                           tgt_key_padding_mask= Ypad_hidden_mask  \
                                          ).permute([1,0,2]) # [batch_size,max_length,output_dim]
        outputs = self.Linear(outputs)

        return F.log_softmax(outputs, dim=-1)
    
    def encode(self,Xword_embeddings,*args,**kwargs):
        # [max_length, batch_size, hidden_dim]
        return self.embedding(Xword_embeddings).permute([1,0,2]) 
    
    
    def decode(self,memory, Xpad_hidden_mask, Yword_embeddings, Ypad_hidden_mask, tgt_mask=None, *args, **kwargs):

        if tgt_mask==None:
            tgt_mask = self._generate_square_subsequenhut_mask(Ypad_hidden_mask.shape[1]).to(self.device)
            
        # [max_length, batch_size, hidden_dim]
        return self.transformer_decoder(tgt=self.embedding(Yword_embeddings).permute([1,0,2]),
                                        memory = memory,
                                        tgt_mask = tgt_mask,
                                        tgt_key_padding_mask= Ypad_hidden_mask,
                                        memory_key_padding_mask = Xpad_hidden_mask)

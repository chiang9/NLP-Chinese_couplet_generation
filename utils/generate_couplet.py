__author__ = "Shihao Lin"
__time__   = "2021/11/30"
__version__= "1.0"

import torch
import sys
sys.path.append('.../model')
from model.fusionDataset import FusionDataset

def greedy_decode(model,bert,tokenizer,
                  sent,glyph2ix,
                  pinyin2ix,pos2ix,
                  device):
    """
    Generate a Couplet Sentence through Greedy Approach
    return in list format e.g.(['1','2','3'])
    """
    bert.to(device)
    model.to(device)
    
    model.eval()
    # Generate encoder input
    Xsents_input_ids,Xsents_token_type_ids, \
    Xsents_attention_mask,Xsents_pinyin_ids,\
    Xsents_glyph_ids,\
    Xsents_pos_ids = FusionDataset.prepare_sequence(sents=[sent],
                                                    tokenizer=tokenizer,
                                                    glyph2ix=glyph2ix,
                                                    pinyin2ix=pinyin2ix,
                                                    pos2ix=pos2ix,
                                                    encode=True,
                                                    skip_error=False,
                                                    device=device)

    Xword_embeddings = bert(Xsents_input_ids, \
                         Xsents_token_type_ids, \
                         Xsents_attention_mask  \
                         )['last_hidden_state'].detach()
    encode_input = {'Xword_embeddings':Xword_embeddings,
                    'Xsents_pinyin_ids':Xsents_pinyin_ids, \
                    'Xsents_glyph_ids':Xsents_glyph_ids,\
                    'Xsents_pos_ids':Xsents_pos_ids,
                    'Xpad_hidden_mask':None}
    # ENCODER 
    memory = model.encode(**encode_input)
    
    ys = []
    for i in range(len(sent)):
        # Generate Decoder Input at each step
        Ysents_input_ids,Ysents_token_type_ids,\
        Ysents_attention_mask,Ysents_pinyin_ids,\
        Ysents_glyph_ids,Ysents_pos_ids,\
        trueY,y_mask_ids\
        = FusionDataset.prepare_sequence(sents=[ys],\
                                         tokenizer=tokenizer,\
                                         glyph2ix=glyph2ix,\
                                         pinyin2ix=pinyin2ix,\
                                         pos2ix=pos2ix,\
                                         encode=False,\
                                         skip_error=False,\
                                         device=device)
        
        Yword_embeddings = bert(Ysents_input_ids,\
                                Ysents_token_type_ids,\
                                Ysents_attention_mask \
                               )['last_hidden_state'].detach()
        
        decode_input = {
                        'memory':memory,\
                        'Xpad_hidden_mask':None,\
                        'Yword_embeddings':Yword_embeddings,\
                        'Ysents_pinyin_ids':Ysents_pinyin_ids, \
                        'Ysents_glyph_ids':Ysents_glyph_ids,\
                        'Ysents_pos_ids':Ysents_pos_ids,\
                        'Ypad_hidden_mask':None,\
                        'tgt_mask':y_mask_ids}
        
        out = model.decode(**decode_input)
        
        # get the latest generate word
        prob = model.Linear(out[-1,:,:])
        
        # FYI, while you implement the beam search
        # add F.log_softmax(out, dim=-1) to it to get acutual log prob
        
        _,next_word = torch.max(prob,dim=1)
        next_word = tokenizer.convert_ids_to_tokens(next_word)[0]
        
        # 不存在 glyph dict的词以及'[SEP]' 和'[PAD]'给转换成 '_'。
        next_word = next_word if next_word in glyph2ix else '_'
        ys.append(next_word)
    return ys
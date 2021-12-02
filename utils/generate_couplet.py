__author__ = "Shihao Lin"
__time__   = "2021/12/2"
__version__= "2.0"

import torch
import sys,os,json
from collections import defaultdict

REPO_PATH = "/".join(os.path.realpath(__file__).split("/")[:-2])
# print(REPO_PATH)
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

from transformers import (BertTokenizer,BertConfig,BertModel)
from model.fusionDataset import FusionDataset
import torch.nn.functional as F
from model.fusion_transformer import Fusion_Anchi_Trans_Decoder, Fusion_Anchi_Transformer, Anchi_Decoder,Anchi_Transformer


def greedy_decode(model,bert,tokenizer,
                  sent,glyph2ix,
                  pinyin2ix,pos2ix,
                  ix2glyph,device):
    """
    Generate a Couplet Sentence through Greedy Approach
    Output dimension is glyph dim.
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
        prob = model.Linear(out[-1,0,:])
        # prob = model.Linear(out)

        # FYI, while you implement the beam search
        # add F.log_softmax(out, dim=-1) to it to get acutual log prob
        _,next_word = torch.max(prob,dim=-1)
        # print(next_word)
        next_word = ix2glyph[next_word.item()]
        
        ys.append(next_word)
    return ys


def beam_search_decode(model,k,bert,tokenizer,
                      sent,glyph2ix,
                      pinyin2ix,pos2ix,
                      ix2glyph,device):
    def get_last_log_prob(sent,memory,model=model,\
                     tokenizer=tokenizer,\
                     glyph2ix=glyph2ix,\
                     pinyin2ix=pinyin2ix,\
                     pos2ix=pos2ix,bert=bert,\
                     encode=False,\
                     skip_error=False,\
                     device=device):
        """
        helper function to get the last word log prob
        return torch([n]), where n is dimension of output
        """
        model.eval()
        
        Ysents_input_ids,Ysents_token_type_ids,\
        Ysents_attention_mask,Ysents_pinyin_ids,\
        Ysents_glyph_ids,Ysents_pos_ids,\
        trueY,y_mask_ids\
        = FusionDataset.prepare_sequence(sents=[sent],\
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
        out = model.Linear(out[-1,0,:])
        # get the latest generate word
        prob = F.log_softmax(out,dim=-1)
        return prob
    
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
    
    sequences = [[list(),0.0]]
    for _ in range(len(sent)):
        all_candidates =list()
        
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            
            last_word_prob = get_last_log_prob(seq,memory)
            
            values, indices = last_word_prob.topk(k)
            for j in range(k):
                w = ix2glyph[indices[j].item()]
                candidate = [seq+[w], score - values[j]]
                all_candidates.append(candidate)
                
#             values, indices = last_word_prob.topk(k+len(sent))
#             for j in range(k+len(sent)):
#                 w = ix2glyph[indices[j].item()]
#                 if len(seq) != 0:
#                     if w in seq:
#                         continue
#                 candidate = [seq+[w], score - values[j]]
#                 all_candidates.append(candidate)

        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda t:t[1])
        
        # select k best
        sequences = ordered[:k]
    return sequences

if __name__ == '__main__':
    """
    python evaluation
    """
    s1 = time.time()
    
    config = BertConfig.from_pretrained('../AnchiBERT')
    tokenizer = BertTokenizer.from_pretrained('../AnchiBERT')
    Anchibert = BertModel.from_pretrained('../AnchiBERT',config=config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    with open('../data/char_map.json','r') as f:
        ix2glyph = defaultdict(lambda : '_')
        ix2glyph[0] = '[PAD]'
        glyph2ix = defaultdict(lambda : 1)
        glyph2ix.update({'[CLS]':0,'[SEP]':0,'[PAD]':0})
        for i, k in enumerate(json.load(f).keys(),2):
            glyph2ix[k] = i
            ix2glyph[i] = k
    with open('../data/pinyin_map.json','r') as f:
        pinyin2ix = defaultdict(lambda : 1)
        pinyin2ix.update({'[CLS]':0,'[SEP]':0,'[PAD]':0})
        for i,k in enumerate(json.load(f).keys(),2):
            pinyin2ix[k] = i
    with open('../data/pos_tags.json','r') as f:
        pos2ix = defaultdict(lambda : 0)
        pos2ix.update(json.load(f))

    with open("../couplet/test/in.txt",encoding='utf8') as f:
        te_in =  [row.strip().split() for row in f.readlines()]
    
    #下联  
    with open("../couplet/test/out.txt",encoding='utf8') as f:
        te_out = [row.strip() for row in f.readlines()]
    
    ###############################################################
    #            Change this part based on model                  #
    ###############################################################
    config = { # for Trans_Decoder
        'max_position_embeddings':50,
        'hidden_size':768,
        'layer_norm_eps':1e-12,
        'hidden_dropout':0.1,
        'nhead':12,
        'num_layers':6, # trainable
        'output_dim':9110,# fixed use glyph dim as output
        'device':device
    }
    # <model_name>_<optim>_<batch_num>_<lr>_<epoch>_<num_layer>_<train_size>
    name = 'anchi_de_Adam_128_0001_10_60_6_51k_new'
    model = Anchi_Decoder(config)
    model.load_state_dict(torch.load(f'../result/{name}.pt'))
    
    ################################################################
    
    
    predicts = []
    for sent in te_in:
        predict = beam_search_decode(model=model,
                                k=2,
                              bert=Anchibert,
                              tokenizer=tokenizer,
                              sent=sent,
                              glyph2ix=glyph2ix,
                              pinyin2ix=pinyin2ix,
                              pos2ix=pos2ix,
                              ix2glyph=ix2glyph,
                                device=device)[0][0]
        predicts.append(''.join(predict))
#     for i, j , k in zip(te_in[:5],predicts[:5],te_out[:5]):
#         print('top:',''.join(i))
#         print('predict:',j)
#         print('gold:',k)
    with open(f'../result/{name}_predict.txt','w') as f:
        for i in predicts:
            f.write(i+'\n')
    print('time:',time.time()-s1)

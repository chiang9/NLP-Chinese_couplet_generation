from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate import bleu
import numpy as np
from collections import Counter,defaultdict
import nltk
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk.lm.models import Laplace
from nltk.lm import Vocabulary
import pickle

from generate_couplet import beam_search_decode
from transformers import (BertTokenizer,BertConfig,BertModel)

import sys,os,torch,json,time
# sys.path.append('.../model')
REPO_PATH = "/".join(os.path.realpath(__file__).split("/")[:-2])
print(REPO_PATH)
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
from model.fusionDataset import FusionDataset
from model.fusion_transformer import Fusion_Anchi_Trans_Decoder, Fusion_Anchi_Transformer, Anchi_Decoder,Anchi_Transformer

def evaluate_pred(gold, pred, perplexity_model = '../result/perplexity_model.pt'):
    """evaluation metrics for BLEU and Perplexity

    Args:
        gold (list): gold standard result
        pred (list): prediction from the model
        perplexity_model (str, optional): model to the perplexity model. Defaults to '/result/perplexity_model.pt'.

    Returns:
        avg_bleu: average bleu score
        perplexity_res: list of all perplexity score
    """
    
    
    gold = [list(x.replace(" ","")) for x in gold]
    pred = [list(x.replace(" ","")) for x in pred]
    
    # bleu score
    smoothfct = SmoothingFunction().method2
    bleu_score = 0
    for i in range(len(gold)):
        bleu_score += bleu([pred[i]], gold[i], smoothing_function = smoothfct)
    avg_bleu = bleu_score / len(gold)
    
    # perplexity
    f = open(perplexity_model, 'rb')
    lm = pickle.load(f)
    f.close()

    test_data = [nltk.bigrams(pad_both_ends(t, n = 2)) for t in pred]
    perplexity_res = [lm.perplexity(data) for data in test_data]
    
    return avg_bleu, perplexity_res


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
    config = { # Fusion_Anchi_Transformer
        'max_position_embeddings':50,
        'hidden_size':768,
        'font_weight_path':'../data/glyph_weight.npy',
        'pinyin_embed_dim':30, # trainable
        'pinyin_path':'../data/pinyin_map.json',
        'tag_size':30,
        'tag_emb_dim':10, # trainable 
        'layer_norm_eps':1e-12, 
        'hidden_dropout':0.1, 
        'nhead':12,
        'num_encoder_layers':5, # trainable
        'num_decoder_layers':6, # trainable
        'output_dim':9110,# fixed use glyph dim as output
        'dim_feedforward': 3072,
        'activation':'relu',
        'trans_dropout':0.1,
        'device':device
    }
    # <model_name>_<optim>_<batch_num>_<lr>_<epoch>_<pinyin_embed_dim>_<tag_emb_dim>_<encoder layer>_<decoder layer>_<train_data_size>
    name = 'fu_anchi_tra_Adam_128_00001_60_30_10_5_6_110k'
    model = Fusion_Anchi_Transformer(config)
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
    
    with open(f'../result/{name}_predict.txt','w') as f:
        for i in predicts:
            f.write(i+'\n')
            
    res = evaluate_pred(te_out, predicts)
    print('result',res)
    print('time:',time.time()-s1)
    with open(f'../result/{name}.txt','w') as f:
        f.write(f'{res[0]}\t{res[1][0]}\t{res[1][1]}')
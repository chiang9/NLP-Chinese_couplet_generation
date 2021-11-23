__author__ = "Shihao Lin"
__time__   = "2021/11/23"
__version__= "1.0"

from tqdm import tqdm
import torch
import jieba
import jieba.posseg as pseg
import paddle
from pypinyin import pinyin, Style

class FusionDataset(Dataset):
    def __init__(self,X,tokenizer,glyph2ix,pinyin2ix,pos2ix=None,Y=None, pos_ids_X=None,pos_ids_Y=None,skip_error=True):
        temp = self.prepare_sequence(X,tokenizer,glyph2ix,pinyin2ix,pos2ix,pos_ids_X,skip_error=skip_error)
        self.x_input_ids= temp[0]
        self.x_token_type_ids= temp[1]
        self.x_attention_mask= temp[2]
        self.x_pinyin_ids= temp[3]
        self.x_glyph_ids= temp[4]
        self.x_pos_ids= temp[5]

        if Y:
            temp = self.prepare_sequence(Y,tokenizer,glyph2ix,pinyin2ix,pos2ix,pos_ids_Y,encode=False,skip_error=skip_error)
            self.y_input_ids= temp[0]
            self.y_token_type_ids= temp[1]
            self.y_attention_mask= temp[2]
            self.y_pinyin_ids= temp[3]
            self.y_glyph_ids= temp[4]
            self.y_pos_ids= temp[5]
        else:
            self.y_input_ids= [0]*len(self.x_input_ids)
            self.y_token_type_ids= [0]*len(self.x_input_ids)
            self.y_attention_mask= [0]*len(self.x_input_ids)
            self.y_pinyin_ids= [0]*len(self.x_input_ids)
            self.y_glyph_ids= [0]*len(self.x_input_ids)
            self.y_pos_ids= [0]*len(self.x_input_ids)
        
    def __len__(self):
        return len(self.x_input_ids)
    
    def __getitem__(self,idx):
        return  self.x_input_ids[idx], \
                self.x_token_type_ids[idx], \
                self.x_attention_mask[idx], \
                self.x_pinyin_ids[idx], \
                self.x_glyph_ids[idx], \
                self.x_pos_ids[idx], \
                self.y_input_ids[idx], \
                self.y_token_type_ids[idx], \
                self.y_attention_mask[idx], \
                self.y_pinyin_ids[idx], \
                self.y_glyph_ids[idx], \
                self.y_pos_ids[idx]
    
    
    @classmethod
    def prepare_sequence(cls,sents, tokenizer, glyph2ix,pinyin2ix,pos2ix=None,unpad_sents_pos_ids=None,encode=True,skip_error=True):
        
        # tranform the wrong Chinese Char in dataset to match Char in pinyin library
        char_correct = {'凉':'凉','裏':'裹','郎':'郎','ㄚ':'丫','—':'一'}
        trans2pinyin = lambda sent: [_[0] for _ in pinyin(sent,style=Style.TONE2, heteronym=False,neutral_tone_with_five=True)]

        
        sents_input_ids= []
        sents_token_type_ids= []
        sents_attention_mask= []
        sents_pinyin_ids= []
        sents_glyph_ids= []
        sents_pos_ids= []
        if not unpad_sents_pos_ids:
            unpad_sents_pos_ids = [None]*len(sents)
            paddle.enable_static()
            jieba.enable_paddle()
            
        max_len = max([len(_) for _ in sents])
        for (sent,sent_pos_ids) in tqdm(zip(sents,unpad_sents_pos_ids)):
            bert_tmp = []
            flag = True
            for i in range(len(sent)):
                if  re.match(r'[\ue2a5-\ue8f0]+',sent[i]):
                    if skip_error:
                        flag = False
                        break
                    bert_tmp.append('[UNK]')
                    sent[i] = '_'
                else:
                    if sent[i] in char_correct:
                        sent[i] = char_correct[sent[i]]
                    bert_tmp.append(sent[i])
            if flag:
                # +2 FOR [CLS] & [SEP] +1 FOR [CLS]
                input_ids= torch.zeros(max_len+2,dtype=torch.int) if encode else torch.zeros(max_len+1,dtype=torch.int)
                token_type_ids= input_ids.clone()
                attention_mask= input_ids.clone()
                pinyin_ids= input_ids.clone()
                glyph_ids= input_ids.clone()
                pos_ids= input_ids.clone()
                                
                token_res = tokenizer(''.join(bert_tmp),return_tensors='pt')
                cur_l = len(sent)
                
                if encode:
                    input_ids[:cur_l+2] = token_res['input_ids']
                    token_type_ids[:cur_l+2] = token_res['token_type_ids']
                    attention_mask[:cur_l+2] = token_res['attention_mask']
                else:
                    input_ids[:cur_l+1] = token_res['input_ids'][0,:cur_l+1]
                    token_type_ids[:cur_l+1] = token_res['token_type_ids'][0,:cur_l+1]
                    attention_mask[:cur_l+1] = token_res['attention_mask'][0,:cur_l+1]
                
                counter = 1
                for res in trans2pinyin(''.join(sent)):
                    if re.match(r'\W\_',res):
                        for c in res:
                            pinyin_ids[counter] = pinyin2ix[c]
                            counter+=1
                    else:
                        pinyin_ids[counter] = pinyin2ix[res]
                        counter+=1
                
                glyph_ids[1:cur_l+1] = torch.tensor([glyph2ix[_] for _ in sent],dtype=torch.int)
                
                if not sent_pos_ids:
                    sent_pos_ids = []
                    for word in pseg.cut(''.join(sent), use_paddle=True):
                        sent_pos_ids.extend([pos2ix[word.flag.lower()]]*len(word.word))
                pos_ids[1:cur_l+1] = torch.tensor(sent_pos_ids)
                
                sents_input_ids.append(input_ids)
                sents_token_type_ids.append(token_type_ids)
                sents_attention_mask.append(attention_mask)
                sents_pinyin_ids.append(pinyin_ids)
                sents_glyph_ids.append(glyph_ids)
                sents_pos_ids.append(pos_ids)
        return sents_input_ids,sents_token_type_ids,sents_attention_mask,sents_pinyin_ids,sents_glyph_ids,sents_pos_ids
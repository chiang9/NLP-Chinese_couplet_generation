__author__ = "Shihao Lin"
__time__   = "2021/11/29"
__version__= "2.0"

from tqdm import tqdm
import torch,re
import jieba
import jieba.posseg as pseg
import paddle
from pypinyin import pinyin, Style
from torch.utils.data import Dataset

class FusionDataset(Dataset):
    def __init__(self,X,tokenizer,glyph2ix,pinyin2ix,
                 pos2ix=None,Y=None, pos_ids_X=None,
                 pos_ids_Y=None,skip_error=True,device=None):
        """
        Notes: set device to current device to load dataset into gpu
        
        input_ids,token_type_ids,attention_mask are related to Bert embedding
        pinyin_ids : related to pinyin embedding
        glyph_ids : related to glyph embedding
        pos_ids : related to pos tag embedding

        y_true_ids: related to true y decoder output id that links to Bert vocab

        Return:
        self.x_input_ids[idx], \
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
        self.y_pos_ids[idx],\
        self.y_true_ids[idx]

        """
        temp = self.prepare_sequence(X,tokenizer,glyph2ix,
                                     pinyin2ix,pos2ix,
                                     pos_ids_X,skip_error=skip_error,
                                     device=device)
        self.x_input_ids= temp[0]
        self.x_token_type_ids= temp[1]
        self.x_attention_mask= temp[2]
        self.x_pinyin_ids= temp[3]
        self.x_glyph_ids= temp[4]
        self.x_pos_ids= temp[5]

        if Y:
            temp = self.prepare_sequence(Y,tokenizer,glyph2ix,
                                         pinyin2ix,pos2ix,
                                         pos_ids_Y,encode=False,
                                         skip_error=skip_error,
                                         device=device)
            self.y_input_ids= temp[0]
            self.y_token_type_ids= temp[1]
            self.y_attention_mask= temp[2]
            self.y_pinyin_ids= temp[3]
            self.y_glyph_ids= temp[4]
            self.y_pos_ids= temp[5]
            self.y_true_ids = temp[6]
            self.y_mask_ids = self._generate_square_subsequenhut_mask(self.y_input_ids.shape[1]).to(device)
        else:
            if device:
                self.y_input_ids= torch.zeros(len(self.x_input_ids), dtype=torch.long).to(device)
                self.y_token_type_ids= torch.zeros(len(self.x_input_ids), dtype=torch.long).to(device)
                self.y_attention_mask= torch.zeros(len(self.x_input_ids), dtype=torch.long).to(device)
                self.y_pinyin_ids= torch.zeros(len(self.x_input_ids), dtype=torch.long).to(device)
                self.y_glyph_ids= torch.zeros(len(self.x_input_ids), dtype=torch.long).to(device)
                self.y_pos_ids= torch.zeros(len(self.x_input_ids), dtype=torch.long).to(device)
                self.y_true_ids= torch.zeros(len(self.x_input_ids), dtype=torch.long).to(device)
                self.y_mask_ids = torch.zeros(0)

            else:
                self.y_input_ids= [0]*len(self.x_input_ids)
                self.y_token_type_ids= [0]*len(self.x_input_ids)
                self.y_attention_mask= [0]*len(self.x_input_ids)
                self.y_pinyin_ids= [0]*len(self.x_input_ids)
                self.y_glyph_ids= [0]*len(self.x_input_ids)
                self.y_pos_ids= [0]*len(self.x_input_ids)
                self.y_true_ids= [0]*len(self.x_input_ids)
                self.y_mask_ids = 0
        
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
                self.y_pos_ids[idx],\
                self.y_true_ids[idx],\
                self.y_mask_ids
    
    @classmethod
    def _generate_square_subsequent_mask(cls, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    @classmethod
    def prepare_sequence(cls,sents, tokenizer, glyph2ix,pinyin2ix,pos2ix=None,
                         unpad_sents_pos_ids=None,encode=True,skip_error=True,
                         device=None
                        ):
        
        # tranform the wrong Chinese Char in dataset to match Char in pinyin library
        char_correct = {'凉':'凉','裏':'裹','郎':'郎','ㄚ':'丫','—':'一'}
        trans2pinyin = lambda sent: [_[0] for _ in pinyin(sent,style=Style.TONE2, heteronym=False,neutral_tone_with_five=True)]

        
        sents_input_ids= []
        sents_token_type_ids= []
        sents_attention_mask= []
        sents_pinyin_ids= []
        sents_glyph_ids= []
        sents_pos_ids= []
        if not encode:
            sents_true_ids=[]
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
                elif sent[i] == '_':
                    bert_tmp.append('[UNK]')
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
                if not encode:
                    true_ids = torch.zeros(max_len+2,dtype=torch.int) if encode else torch.zeros(max_len+1,dtype=torch.long)
                                
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
                    true_ids[:cur_l+1] = token_res['input_ids'][0,1:cur_l+2]
                
                counter = 1
                for res in trans2pinyin(''.join(sent)):
                    if '_' in res:
                        for c in res:
#                             try:
                            pinyin_ids[counter] = pinyin2ix[c]
                            counter+=1
#                             except:
#                                 print(bert_tmp,sent,res)
#                                 raise
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
                if not encode:
                    sents_true_ids.append(true_ids)
        if device:
            sents_input_ids = torch.stack(sents_input_ids).to(device)
            sents_token_type_ids= torch.stack(sents_token_type_ids).to(device)
            sents_attention_mask= torch.stack(sents_attention_mask).to(device)
            sents_pinyin_ids= torch.stack(sents_pinyin_ids).to(device)
            sents_glyph_ids= torch.stack(sents_glyph_ids).to(device)
            sents_pos_ids= torch.stack(sents_pos_ids).to(device)
            if not encode:
                sents_true_ids = torch.stack(sents_true_ids).to(device)
        if encode:
            return sents_input_ids,sents_token_type_ids, \
                    sents_attention_mask,sents_pinyin_ids, \
                    sents_glyph_ids,sents_pos_ids
        else:
            return sents_input_ids,sents_token_type_ids, \
                    sents_attention_mask,sents_pinyin_ids, \
                    sents_glyph_ids,sents_pos_ids, sents_true_ids
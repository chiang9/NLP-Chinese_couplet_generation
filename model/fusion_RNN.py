__author__ = "Joe Chen"
__time__   = "2021/11/26"
__version__= "1.0"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


"""    
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    fusion = FusionEmbedding(CONFIG)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT,fusion)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    model = Seq2Seq(enc, dec, device).to(device)
"""

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device
        
    def create_mask(self, src):
        mask = (src != self.src_pad_idx).permute(1, 0)
        return mask
        
    def forward(self, src, src_len, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #src_len = [batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time        
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, src_len)    
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        mask = self.create_mask(src)
        #mask = [batch size, src len]
        for t in range(1, trg_len):
            #insert input token embedding, previous hidden state, all encoder hidden states 
            #  and mask
            #receive output tensor (predictions) and new hidden state
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
            
        return outputs


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
        embeddings = self.LayerNorm(fusion_embed)
        return self.dropout(embeddings)

    
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout,FusionEmbedding):
        super().__init__()
        
        self.embedding = FusionEmbedding
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_len):
        
        #src = [src len, batch size]
        #src_len = [batch size]
        embedded = self.dropout(self.embedding(src))
        #embedded = [src len, batch size, emb dim]        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)       
        packed_outputs, hidden = self.rnn(packed_embedded)                        
        #packed_outputs is a packed sequence containing all hidden states
        #hidden is now from the final non-padded element in the batch    
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)  
        #outputs is now a non-packed sequence, all hidden states obtained
        #  when the input is a pad token are all zeros  
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        
        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]
        
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs, mask):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        #energy = [batch size, src len, dec hid dim
        attention = self.v(energy).squeeze(2)
        #attention = [batch size, src len]
        attention = attention.masked_fill(mask == 0, -1e10) # !!!
        
        return F.softmax(attention, dim = 1)

    
    
    
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs, mask):
             
        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        #mask = [batch size, src len]
        
        input = input.unsqueeze(0)
        #input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        #embedded = [1, batch size, emb dim]
        a = self.attention(hidden, encoder_outputs, mask)       
        #a = [batch size, src len]
        a = a.unsqueeze(1)
        #a = [batch size, 1, src len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        weighted = torch.bmm(a, encoder_outputs)
        #weighted = [batch size, 1, enc hid dim * 2]
        weighted = weighted.permute(1, 0, 2)
        #weighted = [1, batch size, enc hid dim * 2]
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]  
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        #prediction = [batch size, output dim]
        
        return prediction, hidden.squeeze(0), a.squeeze(1)



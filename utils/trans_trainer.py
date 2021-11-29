__author__ = "Shihao Lin"
__time__   = "2021/11/29"
__version__= "1.0"

import os,time,pickle
import torch.optim as optim
import torch
import torch.nn as nn
from copy import deepcopy
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


torch.manual_seed(1)

def train_epoch(model,dataLoader,optimizer,loss_function,
                bert, device,with_trans=True):
    """Train one epoch of the model"""
    epoch_loss = 0    
    model.train()
    
    for Xsents_input_ids,Xsents_token_type_ids,\
        Xsents_attention_mask,Xsents_pinyin_ids,\
        Xsents_glyph_ids,Xsents_pos_ids,\
        Ysents_input_ids,Ysents_token_type_ids,\
        Ysents_attention_mask,Ysents_pinyin_ids,\
        Ysents_glyph_ids,Ysents_pos_ids,trueY,y_mask_ids in dataLoader:
        
#         Xsents_input_ids = Xsents_input_ids.to(device)
#         Xsents_token_type_ids = Xsents_token_type_ids.to(device)
#         Xsents_attention_mask = Xsents_attention_mask.to(device)
#         Xsents_pinyin_ids = Xsents_pinyin_ids.to(device)
#         Xsents_glyph_ids = Xsents_glyph_ids.to(device)
#         Xsents_pos_ids = Xsents_pos_ids.to(device)
#         Ysents_input_ids = Ysents_input_ids.to(device)
#         Ysents_token_type_ids = Ysents_token_type_ids.to(device)
#         Ysents_attention_mask = Ysents_attention_mask.to(device)
#         Ysents_pinyin_ids = Ysents_pinyin_ids.to(device)
#         Ysents_glyph_ids = Ysents_glyph_ids.to(device)
#         Ysents_pos_ids = Ysents_pos_ids.to(device)
        
        Xword_embeddings = bert(Xsents_input_ids,      \
                                 Xsents_token_type_ids, \
                                 Xsents_attention_mask  \
                                 )['last_hidden_state'].detach()


        Yword_embeddings = bert(Ysents_input_ids,      \
                                Ysents_token_type_ids, \
                                Ysents_attention_mask  \
                               )['last_hidden_state'].detach()
        
        inputs = {'Xword_embeddings':Xword_embeddings, \
                  'Xsents_pinyin_ids':Xsents_pinyin_ids, \
                  'Xsents_glyph_ids':Xsents_glyph_ids,\
                  'Xsents_pos_ids':Xsents_pos_ids,\
                  'Yword_embeddings':Yword_embeddings,\
                  'Ysents_pinyin_ids':Ysents_pinyin_ids, \
                  'Ysents_glyph_ids':Ysents_glyph_ids,\
                  'Ysents_pos_ids':Ysents_pos_ids, \
                  'device':device}
        
        if with_trans:
            inputs['Xpad_hidden_mask'] = (~ Xsents_attention_mask.bool()).detach()
            inputs['Ypad_hidden_mask'] = (~ Ysents_attention_mask.bool()).detach()
            inputs['tgt_mask']= y_mask_ids[0]
        
        # Clear gradients
        optimizer.zero_grad()
            
        outputs = model(**inputs) # [batch_size,max_length,output_dim]
        outputs = outputs.view(-1,outputs.shape[-1])   #[batch_size*max_length,output_dim]
                
        trueY = trueY.view(-1)
        
        # Calculate the loss
        loss = loss_function(outputs,trueY)
        
        # with torch.autograd.detect_anomaly():
        
        # Get gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        epoch_loss += loss.item()
    return epoch_loss / len(dataLoader)

def evaluate_epoch(model,dataLoader,loss_function,
                   bert, device,with_trans=True):
    """Evaluate one epoch of the model"""
    epoch_loss = 0    
    model.eval()
    with torch.no_grad():
         for Xsents_input_ids,Xsents_token_type_ids,\
            Xsents_attention_mask,Xsents_pinyin_ids,\
            Xsents_glyph_ids,Xsents_pos_ids,\
            Ysents_input_ids,Ysents_token_type_ids,\
            Ysents_attention_mask,Ysents_pinyin_ids,\
            Ysents_glyph_ids,Ysents_pos_ids,trueY,y_mask_ids in dataLoader:

#             Xsents_input_ids = Xsents_input_ids.to(device)
#             Xsents_token_type_ids = Xsents_token_type_ids.to(device)
#             Xsents_attention_mask = Xsents_attention_mask.to(device)
#             Xsents_pinyin_ids = Xsents_pinyin_ids.to(device)
#             Xsents_glyph_ids = Xsents_glyph_ids.to(device)
#             Xsents_pos_ids = Xsents_pos_ids.to(device)
#             Ysents_input_ids = Ysents_input_ids.to(device)
#             Ysents_token_type_ids = Ysents_token_type_ids.to(device)
#             Ysents_attention_mask = Ysents_attention_mask.to(device)
#             Ysents_pinyin_ids = Ysents_pinyin_ids.to(device)
#             Ysents_glyph_ids = Ysents_glyph_ids.to(device)
#             Ysents_pos_ids = Ysents_pos_ids.to(device)

            Xword_embeddings = bert(Xsents_input_ids,      \
                                     Xsents_token_type_ids, \
                                     Xsents_attention_mask  \
                                     )['last_hidden_state'].detach()


            Yword_embeddings = bert(Ysents_input_ids,      \
                                    Ysents_token_type_ids, \
                                    Ysents_attention_mask  \
                                   )['last_hidden_state'].detach()

            inputs = {'Xword_embeddings':Xword_embeddings, \
                      'Xsents_pinyin_ids':Xsents_pinyin_ids, \
                      'Xsents_glyph_ids':Xsents_glyph_ids,\
                      'Xsents_pos_ids':Xsents_pos_ids,\
                      'Yword_embeddings':Yword_embeddings,\
                      'Ysents_pinyin_ids':Ysents_pinyin_ids, \
                      'Ysents_glyph_ids':Ysents_glyph_ids,\
                      'Ysents_pos_ids':Ysents_pos_ids, \
                      'device':device}

            if with_trans:
                inputs['Xpad_hidden_mask'] = (~ Xsents_attention_mask.bool()).detach()
                inputs['Ypad_hidden_mask'] = (~ Ysents_attention_mask.bool()).detach()
                inputs['tgt_mask']= y_mask_ids[0]


            outputs = model(**inputs) # [batch_size,max_length,output_dim]
            outputs = outputs.view(-1,outputs.shape[-1])   #[batch_size*max_length,output_dim]

            trueY = trueY.view(-1)

            # Calculate the loss
            loss = loss_function(outputs,trueY)
            epoch_loss += loss.item()
    return epoch_loss / len(dataLoader)

def timeparser(elapse):
    mins = int(elapse/60)
    sec = int(elapse - (mins*60))
    return mins, sec

def train(model,trainSet,validSet,batch_size,lr, epoch,bert,name= None,
          with_trans= True,optimizer_name= 'Adam',
          scheduleFactor=0.5,schedule_Patience=5,
          min_lr=1e-06, verbose=True, patience= 10,store='../result/'):
    """
    Training a given neural network model
    Return a Best model object in state_dict() fashion
    
    @Para:
    model: pytorch training model
    trainSet: Tensor Dataset
    validSet: Tensor Dataset
    batch_size: batch_size
    lr: initial learning rate
    epoch: num of epoch
    bert: transformers.models.bert.modeling_bert.BertModel
    name: Model name
    with_trans: If True, a transformer padding mask will be generated as forward input. 
    
    if_writer: True for using tensorboard to trail
        the train loss and the valid loss at each epoch
    
    scheduleFactor: reduce factor for learning rate
    schedule_Patience:(int): Number of epochs with no improvement after
        which learning rate will be reduced. For example, if
        `patience = 2`, then we will ignore the first 2 epochs
        with no improvement, and will only decrease the LR after the
        3rd epoch if the loss still hasn't improved then.
        Default: 10.
    
    min_lr: min_lr
    verbose: True for print train loss and valid loss at each epoch
    patience:  Number of epochs with no improvement after
        which training will be stopped. For example, if
        `patience = 2`, then we will ignore the first 2 epochs
        with no improvement, and will only stops after the
        3rd epoch if the valid loss still hasn't improved then.
        Default: 5.
    store: Place to store output model and losses
    """
    if not os.path.exists(store):
        os.mkdir(store)
        print(store,'is created.')
        
     # Instantiate Train Loader and Valid Loader
    trainLoader = DataLoader(trainSet,batch_size=batch_size,shuffle=True)
    validLoader = DataLoader(validSet,batch_size=batch_size,shuffle=False)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    bert.to(device)

    # Instantiate loss class
    criterion = nn.NLLLoss(ignore_index=0)
    
    # Iniantiate optimizer class
    lr = lr
    
    # Iniantiate optimizer class
    if optimizer_name:
        optimizer = optim.Adam(model.parameters(),lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(),lr = lr, momentum=.9, nesterov=True)

     # Reduce on Loss Plateau Learning rate scheduler

    scheduler = ReduceLROnPlateau(optimizer=optimizer,factor=scheduleFactor,patience=schedule_Patience, min_lr=min_lr,verbose=verbose)
    
    # Instantiate Best model and Best Valid Loss
    best_valid_loss = float('inf')
    
    valid_patience_counter = 0
    
    n_epoch = epoch
    train_losses = []
    valid_losses = []
    for epoch in tqdm(range(n_epoch)):
        
        start_time = time.time()
        
        # train model (average)
        train_loss = train_epoch(model,trainLoader,optimizer,
                                 criterion,bert,device,with_trans)
        
        # Decay Learning Rate
        scheduler.step(train_loss)
        
        ##############
        # validation #
        ##############
        valid_loss = evaluate_epoch(model,validLoader,criterion,
                                               bert,device,with_trans)
        mins,secs = timeparser(time.time()-start_time)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        
        if verbose:
            # Print Learning Rate and Loss for ecach Loss
            print(f'Epoch: {epoch+1:02} | Epoch Time: {mins}m {secs}s')
            print(f'\tTraining Loss: {train_loss:.5f} \tValidation Loss: {valid_loss:.5f}')
        
        # Compare the Valid Loss with Best Valid Loss
        if valid_loss< best_valid_loss:
            # reset counter
            valid_patience_counter = 0
            best_model = deepcopy(model.state_dict())
            torch.save(best_model,os.path.join(store,f'{name}.pt'))
            with open(os.path.join(store,f'{name}_losses.pt'),'wb') as f:
                pickle.dump((train_losses,valid_losses),f)
            best_valid_loss = valid_loss
            
        else:
            valid_patience_counter += 1
            if valid_patience_counter == patience:
                break
#                 return best_model,train_losses,valid_losses
    torch.save(best_model,os.path.join(store,f'{name}.pt'))
    
    with open(os.path.join(store,f'{name}_losses.pt'),'wb') as f:
        pickle.dump((train_losses,valid_losses),f)
#     return best_model, train_losses,valid_losses
## Acknowledge
# - https://towardsdatascience.com/3-types-of-contextualized-word-embeddings-from-bert-using-transfer-learning-81fcefe3fe6d
# - https://zhuanlan.zhihu.com/p/113639892

import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
import pandas as pd

class preprocessing(object):
    def __init__(self):
        pass
    

    def bert_text_preparation(self, text, tokenizer):
        """Preparing the input for BERT
        
        Takes a string argument and performs
        pre-processing like adding special tokens,
        tokenization, tokens to ids, and tokens to
        segment ids. All tokens are mapped to seg-
        ment id = 1.
        
        Args:
            text (str): Text to be converted
            tokenizer (obj): Tokenizer object
                to convert text into BERT-re-
                adable tokens and ids
            
        Returns:
            list: List of BERT-readable tokens
            obj: Torch tensor with token ids
            obj: Torch tensor segment ids
        
        
        """
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = tokenizer.tokenize(marked_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1]*len(indexed_tokens)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        return tokenized_text, tokens_tensor, segments_tensors

    def get_bert_embeddings(self, tokens_tensor, segments_tensors, model):
        """Get embeddings from an embedding model
        
        Args:
            tokens_tensor (obj): Torch tensor size [n_tokens]
                with token ids for each token in text
            segments_tensors (obj): Torch tensor size [n_tokens]
                with segment ids for each token in text
            model (obj): Embedding model to generate embeddings
                from token and segment ids
        
        Returns:
            list: List of list of floats of size
                [n_tokens, n_embedding_dimensions]
                containing embeddings for each token
        
        """
        
        # Gradient calculation id disabled
        # Model is in inference mode
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
            # Removing the first hidden state
            # The first state is the input state
            hidden_states = outputs[2][1:]

        # Getting embeddings from the final BERT layer
        token_embeddings = hidden_states[-1]
        # Collapsing the tensor into 1-dimension
        token_embeddings = torch.squeeze(token_embeddings, dim=0)
        # Converting torchtensors to lists
        list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

        return list_token_embeddings



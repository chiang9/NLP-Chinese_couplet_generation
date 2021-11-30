from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate import bleu
import numpy as np
from collections import Counter
import nltk
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk.lm.models import Laplace
from nltk.lm import Vocabulary
import pickle

def evaluate_pred(gold, pred, perplexity_model = '/result/perplexity_model.pt'):
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
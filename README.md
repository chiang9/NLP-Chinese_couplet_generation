# NLP-Chinese_couplet_generation
 
### Way to generate data/glyph_weight.npy
```
python utils/glyph_generator.py
```

### Naming Convention for Transfomer model result
- `Fusion_Anchi_Trans_Decoder`
```
# <model_name>_<optim>_<batch_num>_<lr>_<epoch>_<num_layer>_<pinyin_embed_dim>_<tag_emb_dim>_<train_set_size>
name = 'fu_anchi_de_Adam_128_01_60_6_30_10_110k'
```
- `Fusion_Anchi_Transformer`
```
# <model_name>_<optim>_<batch_num>_<lr>_<epoch>_<pinyin_embed_dim>_<tag_emb_dim>_<encoder layer>_<decoder layer>_<train_set_size>
name = 'fu_anchi_tra_Adam_128_01_60_30_10_5_6_110k'
```
- `Anchi Decoder`
```
# <model_name>_<optim>_<batch_num>_<lr>_<epoch>_<num_layer>_<pinyin_embed_dim>_<tag_emb_dim>_<train_set_size>
name = 'anchi_de_Adam_128_01_10_60_6_110k'
```
- `Anchi Transformer`
```
# <model_name>_<optim>_<batch_num>_<lr>_<epoch>_<encoder layer>_<decoder layer>_<train_set_size>
name = 'anchi_tra_Adam_128_01_60_5_6_110k'
```

### Main Update
1. `model/fusion_transformer.py` contains:  
    - `Fusion_Anchi_Trans_Decoder`:  
        - Fusion Layer (Pinyin + Glyph + Pos + Anchi Bert last hidden layer) as Multi-Head Layer's memory
    and Output Embedding to Transformer Decoder  
    - `Fusion_Anchi_Transformer`:  
        - Transformer with Fusion Layer (Pinyin + Glyph + Pos + Anchi Bert last hidden layer) as Input Embedding and Output Embedding  
2. `Demo.ipynb` contains:
    - Examples for model training
    - Examples for transforming raw sentence to model input
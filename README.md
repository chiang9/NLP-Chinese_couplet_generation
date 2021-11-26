# NLP-Chinese_couplet_generation
 
### Way to generate data/glyph_weight.npy
```
python utils/glyph_generator.py
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
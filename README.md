# NLP-Chinese_couplet_generation

### Introduction

Chinese couplet is a special form of poetry composed of complex syntax with ancient Chinese language. Due to the complexity of semantic and grammatical rules, creation of a suitable couplet is a formidable challenge. This paper presents an transformer-based sequence-to-sequence automatic couplet generation model. With the utilization of PinYin and Part-of-Speech tagging, the model achieves the couplet generation. Moreover, we evaluate the [AnchiBERT](https://arxiv.org/abs/2009.11473) on the ancient Chinese language understanding to further improve the model. 

Arxiv Link: https://arxiv.org/abs/2112.01707

### Installation

#### Generate pretrained model file
```
python utils/glyph_generator.py
```

### Sample Result

|                   | 瑶   池   嫩   叶   初   呈   瑞 |
|-------------------|------------|
| Gold              | 玉   树   新   枝   正   发   荣     |
| Anchi Decoder     | 玉   井   香   花   早   有   芳     |
| Anchi Transformer | 紫   井   金   花   尽   上   香     |
| Fusion Decoder    | 宝   井   红   花   正   有   香     |
| Fusion Transformer| 玉   井   红   花   永   有   春     |


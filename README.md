# NLP-Chinese_couplet_generation
 
##Task1 Data Loading @Joe
### Summary (note from Joe)

- idea0: word2vec的参数因为时间关系我没怎么调，后续可以再优化。大家可以找找有没有别的更好的中文embedding的现有模型或者package能用。

- idea1: 现在分词没有研究，因为数据集里每个字中间都有空格。我assume可以用单字来做。可以探索一下别的中文分词模型，看看能不能分出来两个字的词，比如‘晚风’而不是‘晚’和‘风’。这样的word embeeding可能会有更好的效果。

- idea2: 因为现在我们improve模型的概率好像不大，所以我想能不能在数据上下功夫。确定分词后，可以尝试用现有算法或者人工标注数据里的一些特征。比如地点，人名，以及形容词（AABB结构之类的）等（之前有paper做过），或者是平仄发音，韵脚押韵（目前没有人做过这个）。

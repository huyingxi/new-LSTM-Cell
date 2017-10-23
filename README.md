# new-LSTM-Cell
《Joint Extraction of Entities and Relations Based on a Novel Tagging Scheme》-  论文实现 by - pytorch 


这是基于pytorch实现的该论文模型

主要在于改变了LSTM内部实现


分为：
LSTMCell_AddC             ->     encoder部分使用的LSTM
LSTMCell_AddC_decoder     ->     decoder部分使用的LSTM


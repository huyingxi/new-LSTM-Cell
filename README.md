# new-LSTM-Cell
《Joint Extraction of Entities and Relations Based on a Novel Tagging Scheme》-  论文实现 by - pytorch 


这是基于pytorch实现的该论文模型

<img src="https://github.com/huyingxi/new-LSTM-Cell/blob/master/WechatIMG92.jpg" />

主要在于改变了LSTM内部实现


分为：

LSTMCell_AddC             ->     encoder部分使用的LSTM

LSTMCell_AddC_decoder     ->     decoder部分使用的LSTM


其中：

lstm-lstm-pytorch-evaluate-embed.py是模型部分

rnn.py是根据需要修改了pytorch的内部实现


使用：

将pytorch下的nn/_function/rnn.py替换即可


...-GPU 是支持GPU运行的版本

...-embed 是本地运行版本


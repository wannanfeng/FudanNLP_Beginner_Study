用LSTM、GRU来训练字符级的语言模型，通过语言模型来生成诗句。

除了给定的各种序列类别之外，还要另外多加3个类别，分别是: < pad >,< start >,< end >，
分别代表padding（即补位，使句子达到同一个长度），句子开头和句子结尾，总共C类标签。

由于数据量较少,采用batch大小为1,因此并无padding, 即padding的部分在本次实战中并不会真正参与到任何的计算
（over fitting了）

对于生成随机诗句
    随机初始化一个向量（长度为lh）
    输入到 LSTM / GRU 中，得到新的向量，这个向量长度为 lh
    输入到全连接层，得到新的向量，长度为 C，代表下一个字在C种字符的得分
    对该向量取最高分（最大值），对应第i个索引（index），就是下一个字 i
    如果下一个字是 “句号” 或者 “< end >”，给该诗句画上句号，开始重复以上步骤生成下一句。

对于生成藏头诗句
    选择一个在词嵌入字典中已有的字，获得其对应的特征向量（长度为 lh）
    输入到 LSTM / GRU 中，得到新的向量，这个向量长度为 lh
    输入到全连接层，得到新的向量，长度为 C ，代表下一个字在C种字符的得分
    对该向量取最高分（最大值），对应第i个索引（index），就是下一个字 i
    如果下一个字是 “句号” 或者 “< end >”，给该诗句画上句号，开始重复以上步骤生成下一句。

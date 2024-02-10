import collections,os,random,urllib
import numpy as np
import pandas as pd
import tensorflow as tf

data1=pd.read_csv('spam_ham_dataset.csv')
ind=data1.loc[data1['label_num']==1].index
content=data1.iloc[ind,2]
print(content)
print('1')
#label=data1.iloc[:,3]


#构建分句表，将大写转成小写
doclist=[]
for i in content:#i是每一封邮件的内容
    i=i.lower()
    for char in '!"#$%&()*+,-./:;<=>?@[\\]^_‘{|}~':
        i=i.replace(char," ")
    doclist.append(i)

all_words=[]
for wordlist in doclist:
    w=wordlist.split()
    for word in w:
        all_words.append(word)#一个所有单词的词表


print('length:',len(all_words))

#基本训练参数
learning_rate=0.05
batch_size=128
num_steps=500000 #训练次数
display_step=10000 #打印间隔
eval_step=20000 #实验次数

#测试样例
eval_words=['monday','of','night','june']

#word2vec params
embedding_size=200 #词向量维数
max_vocabulary_size=50000 #语料库词语数
min_occurence=5 #最小词频
skip_window=3 #左右窗口大小
num_skips=2 #一次制作多少个输入输出对
num_sampled=64 #负采样

#创建计数器，计算每个词出现次数
count=[('unk',-1)]#unknow 的词
#返回最常见的max_vocabulary_size个词
count.extend(collections.Counter(all_words).most_common(max_vocabulary_size-1))
#print('max common:',count[0:10])

#将出现次数低于min_occurence 的词删除
for i in range(len(count)-1,-1,-1):#从末尾开始遍历(低频到高频遍历）
    if count[i][1] < min_occurence:
        count.pop(i)
    else:
        break #剩下的都大于min_occurence，符合要求

#词-id映射
vocabulary_size=len(count)
#每个词分配一个id
word2id=dict()
for i ,(word,_) in enumerate(count):
    word2id[word]=i
#print('id:',word2id)

#将原文的词与索引对应起来
data=list()
unk_count=0#未知的词（unknow）默认为0
for word in all_words:
    index=word2id.get(word,0)#UNK索引默认为0
    if index==0:
        unk_count+=1
    data.append(index)
count[0]=('unk',unk_count)
id2word=dict(zip(word2id.values(),word2id.keys()))

print('words count:',len(all_words))
print('unique words',len(set(all_words)))
print('vocabulary size',vocabulary_size)
print('most common words:',count[:10])

#构建所需要的训练数据
data_index=0

#中间词---》（预测）上下文词
def next_batch(batch_size,num_skips,skip_window):
    global data_index
    assert batch_size % num_skips ==0
    assert num_skips<=2* skip_window
    batch= np.ndarray(shape=(batch_size),dtype=np.int32)
    labels=np.ndarray(shape=(batch_size,1),dtype=np.int32)
    #get the  window size
    span=2*skip_window+1 # 左3右3预测中间1
    buffer=collections.deque(maxlen=span)#创建一个长度为7的队列
    if data_index +span >len(data):#如果数据窗口被滑完一遍了
        data_index=0
    buffer.extend(data[data_index:data_index+span])#队列默认存储的是当前窗口：如[43,23,12,6.123.232.56]
    data_index+=span
    for i in range(batch_size//num_skips):#表示每次取多少组不同的词作为输出#每次循环产生batch_size//num_steps对数据，如【21-》11,12,13】，【26-》11,12,13】
        context_words=[w for w in range(span) if w !=skip_window]#上下文是[0,1,2,3,4,5,6]
        words_to_use=random.sample(context_words,num_skips)#在上下文中随机选择两个词作为预测输入
        for j ,context_words in enumerate(words_to_use):
            batch[i*num_skips+j]=buffer[skip_window]#当前窗口中间词
            labels[i*num_skips+j,0]=buffer[context_words]#当前候选词当作标签

        if data_index==len(data):#索引已经到达数据尽头
            buffer.extend(data[0:span])#队列重新取数据的首span个
            data_index=span
        else:
            buffer.append(data[data_index])#之前已传入7个词，窗口右移（队列要右移）,首个元素自动剔除
            data_index+=1

    data_index=(data_index+len(data)-span)% len(data)
    return batch,labels

embedding=tf.Variable(tf.random.normal([vocabulary_size,embedding_size]))#随机初始化
nce_weights=tf.Variable(tf.random.normal([vocabulary_size,embedding_size]))
nce_biases=tf.Variable(tf.zeros([vocabulary_size]))
#索引转化成词向量
def get_embedding(x):
    x_embed=tf.nn.embedding_lookup(embedding,x)
    return x_embed

#loss function:正样本和 负样本对应的output and label
#sigmoid cross entropy
def nce_loss(x_embed,y):
    y=tf.cast(y,tf.int64)
    loss=tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=y,
                       inputs=x_embed,
                       num_sampled=num_sampled,#负采样个数
                       num_classes=vocabulary_size))
    return loss

#测试观察模块
def evaluate(x_embed):
    x_embed=tf.cast(x_embed,tf.float32)
    x_embed_norm=x_embed/tf.sqrt(tf.reduce_sum(tf.square(x_embed)))#归一化
    embedding_norm=embedding/tf.sqrt(tf.reduce_sum(tf.square(embedding),1,keepdims=True),tf.float32)
    cosine_sim_op=tf.matmul(x_embed_norm,embedding_norm,transpose_b=True)#余弦相似度
    return cosine_sim_op

#迭代优化
optimizer=tf.optimizers.SGD(learning_rate)
def run_optimization(x,y):
    with tf.GradientTape() as g:
        emb=get_embedding(x)
        loss=nce_loss(emb,y)

    #计算梯度
    gradients=g.gradient(loss,[embedding,nce_weights,nce_biases])
    #更新
    optimizer.apply_gradients(zip(gradients,[embedding,nce_weights,nce_biases]))

#待测试的几个词
x_test=np.array([word2id[w] for w in eval_words])

#训练
for step in range(1,num_steps+1):
    batch_x,batch_y=next_batch(batch_size,num_skips,skip_window)
    run_optimization(batch_x,batch_y)

    if step % display_step==0 or step==1:
        print('evaluation:')
        sim=evaluate(get_embedding(x_test)).numpy()
        for i in range(len(eval_words)):
            top_k=8#返回前8个最相似的
            nearest=(-sim[i,:]).argsort()[1:top_k+1]
            log_str='%s neighbours:'% eval_words[i]
            for k in range(top_k):
                log_str="%s  %s," % (log_str,id2word[nearest[k]])
            print(log_str)





data=[
    ['r','z','h','j','p'],
    ['z','y','x','w','v','u','t','s'],
    ['z'],
    ['r','x','n','o','s'],
    ['y','r','x','z','q','t','p'],
    ['y','z','x','e','q','s','t','m']
]


class treeNode:
    def __init__(self,name,num,paraentNode):
        self.name=name#节点名字
        self.count=num #频数
        self.nodeLink=None  #链接相似元素项
        self.parent=paraentNode#当前父节点
        self.children={} #存放子节点

    #调整计数
    def inc(self,num):
        self.count+=num
    #spcae :空格个数；同一级的叶子节点缩进个数相同
    def disp(self,space=1):
        print(" "*space,self.name," ",self.count)
        for child in self.children.values():
            #子节点向右缩减
            child.disp(space+1)

#构建字典统计每个元素的频数
def createInitSet(data):
    retDict={}
    for trans in data:
        fset=frozenset(trans)
        retDict.setdefault(fset,0)
        retDict[fset]+=1
    return retDict

#更新头指针表
def updateHeader(nodeToTest,targetNode):
    #遍历，直至到达链表尾部
    while nodeToTest.nodeLink !=None:
        nodeToTest=nodeToTest.nodeLink
    #将节点连上新的节点
    nodeToTest.nodeLink=targetNode

"""
items: 一条新的交易记录
haedTable:头指针表
count:items[0]出现频数
"""
#headTable字典 格式： key: z  values: [计数，链表连接的下一个节点]
def updateTree(items,myTree,headTable,count):
    #先遍历items的第一个元素
    #看它是否为树的子节点；若是，则增加计数；否则，创建一个新的子节点
    if items[0] in myTree.children:
        myTree.children[items[0]].inc(count)
    #现在myTree 为一个空节点
    else:
        myTree.children[items[0]]=treeNode(items[0],count,myTree)
        #如果headTable对应元素还没有连接到别的元素，则先将headTable与新加入的元素连接
        if headTable[items[0]][1]==None:
            headTable[items[0]][1] =myTree.children[items[0]]
        else:
            # 如果headTable有连接到别的元素，则将headTable连接的元素与新加入的元素连接
            updateHeader(headTable[items[0]][1],myTree.children[items[0]])
    #继续遍历，从第二个元素开始;此时遍历items[0]开始的子树
    if len(items)>1:
        updateTree(items[1:],myTree.children[items[0]],headTable,count)


def createTree(data,minSup):
    headTable={}
    head={}
    #第一次遍历数据集，记录每个项的支持度（频数）
    for trans in data:
        for item in trans:
            head[item]=head.get(item,0)+1
    #根据最小支持度过滤
    lessThanMinsup=list(filter(lambda k:head[k]<minSup,head.keys()))
    for k in lessThanMinsup:
        del (head[k])
    freqItemSet=set(head.keys())
    #如果所有数据都不满足最小支持度
    if len(freqItemSet)==0:
        return None,None
    #初始化headerTable(带指针）
    for k in head:
        headTable[k]=[head[k],None]
    #初始化FP-tree
    myTree=treeNode('empty',1,None)
    #第二次遍历数据集，构建FP-tree
    for tranSet,count in data.items():
        #对每一条交易记录的每一项进行遍历
        localD={}
        for item in tranSet:
            #筛选出每条交易记录中在频繁项中的元素,并记录其频数
            if item in freqItemSet:
                localD[item]=headTable[item][0]
        #先根据头指针表的频数，再根据字母顺序进行排序(对频繁项集排序)
        if len(localD)>0:
            orderItems=[v[0] for v in sorted(localD.items(),key=lambda p:(p[1],p[0]),reverse=True)]
            updateTree(orderItems,myTree,headTable,count)
    return myTree,headTable

dataset=createInitSet(data)
Tree,headTable=createTree(dataset,3)

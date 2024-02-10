#生成一些二维数据用于DBSCAN聚类
from sklearn.datasets import make_moons
import pandas as pd
x,y=make_moons(200,noise=0.05,random_state=0)
data=pd.DataFrame(x)
data['label']=y

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import time
# 计算距离矩阵
def compute_squared_EDM(X):
  return squareform(pdist(X,metric='euclidean'))

# DBSCAN算法核心过程
def DBSCAN(data,eps,minPts):
    # 获得距离矩阵
    disMat = compute_squared_EDM(data)
    # 获得数据的行和列(一共有n条数据)
    n, m = data.shape
    # 将矩阵的中小于minPts的数赋予1，大于minPts的数赋予零，然后1代表对每一行求和,然后求核心点坐标的索引
    core_points_index = np.where(np.sum(np.where(disMat <= eps, 1, 0), axis=1) >= minPts)[0]
    # 初始化类别，-1代表未分类。
    labels = np.full((n,), -1)
    clusterId = 0
    # 遍历所有的核心点
    for pointId in core_points_index:
      
        # 如果核心点未被分类，将其作为的种子点，开始寻找相应簇集
        if (labels[pointId] == -1):
            # 首先将点pointId标记为当前类别(即标识为已操作)
            labels[pointId] = clusterId
            # 然后寻找种子点的eps邻域且没有被分类的点，将其放入种子集合
            neighbour=np.where((disMat[:, pointId] <= eps) & (labels==-1))[0]
            seeds = set(neighbour)
            # 通过种子点，开始生长，寻找密度可达的数据点，一直到种子集合为空，一个簇集寻找完毕
            while len(seeds) > 0:
                # 弹出一个新种子点
                newPoint = seeds.pop()
                # 将newPoint标记为当前类
                labels[newPoint] = clusterId
                # 寻找newPoint种子点eps邻域（包含自己）
                queryResults = np.where(disMat[:,newPoint]<=eps)[0]
                # 如果newPoint属于核心点，那么newPoint是可以扩展的，即密度是可以通过newPoint继续密度可达的
                if len(queryResults) >= minPts:
                    # 将邻域内且没有被分类的点压入种子集合
                    for resultPoint in queryResults:
                        if labels[resultPoint] == -1:
                            seeds.add(resultPoint)
            # 簇集生长完毕，寻找到一个类别
            clusterId = clusterId + 1
    return labels


label=DBSCAN(x,0.3,15)

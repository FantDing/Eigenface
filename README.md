# 简介
使用PCA，完成人脸数据特征脸的显示

# 思路
## 数据收集
> 仓库中的数据集仅为了方便下载，侵删

- URL： https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
- 数据组织：共有40个人，每个人有10张图片，分别放在`s0-s40`文件夹下
## 流程
- 每个人选取一张图片，并分别`flatten`成列向量，在`axis=1`方向叠加形成输入数据X
    - X.shape: (10304,40)
- PCA(X)
    - 每个维度去均值化处理
    - 计算协方差矩阵`C=X'.X`
        - 注意不是`X.X'`，所以后面计算出来的特征向量`eig_value`需要进行变换。参考[wiki](https://zh.wikipedia.org/wiki/%E7%89%B9%E5%BE%81%E8%84%B8 )
    - 计算出`C`的特征值，特征向量
        - 使用`numpy.linalg.eigh(C)`会直接将特征值、特征向量排序。需要再进行倒序排列
        - 使用C计算出来的特征值，需要再处理参考[wiki](https://zh.wikipedia.org/wiki/%E7%89%B9%E5%BE%81%E8%84%B8 )
        
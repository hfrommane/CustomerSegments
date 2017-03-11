# coding: UTF-8
import numpy as np
import pandas as pd
import visuals as vs
from IPython.display import display  # 使得我们可以对DataFrame使用display()函数
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier

# 载入整个客户数据集
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis=1, inplace=True)
    print("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))
except:
    print("Dataset could not be loaded. Is the dataset missing?")

# 分析数据
# 显示数据集的一个描述
# Fresh:生鲜
# Milk:牛奶
# Grocery:食品杂货
# Frozen:冷冻
# Detergents:清洁剂
# Delicatessen:熟食
# display(data.describe())

# 练习: 选择样本
# 从数据集中选择三个你希望抽样的数据点的索引
indices = [35, 39, 86]

# 为选择的样本建立一个DataFrame
samples = pd.DataFrame(data.loc[indices], columns=data.keys()).reset_index(drop=True)
# print("Chosen samples of wholesale customers dataset:")
# display(samples)

# 0号样本，杂货需求量接近75%，清洁纸大于75%，可能是一家便利店
# 1号样本，生鲜和冷冻需求量大于75%，可能是一家餐厅
# 2号样本，生鲜、牛奶、杂货、清洁纸需求量都大于75%，可能是一家综合超市

# 练习: 特征相关性
# 为DataFrame创建一个副本，用'drop'函数丢弃一些指定的特征
new_data = data.drop(['Detergents_Paper'], axis=1)
new_label = data['Detergents_Paper']

# 使用给定的特征作为目标，将数据分割成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(new_data, new_label, test_size=0.25, random_state=0)

# 创建一个DecisionTreeRegressor（决策树回归器）并在训练集上训练它
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)

# 输出在测试集上的预测得分
score = regressor.score(X_test, y_test)
# print('Detergents_Paper: {:.4f}'.format(score))
# Fresh: -0.2525
# Milk: 0.3657
# Grocery: 0.6028
# Frozen: 0.2540
# Detergents_Paper: 0.7287
# Delicatessen: -11.6637

# 我选择'Detergents_Paper'。
# 报告的分数为0.729。
# 我认为这个功能不是必要的，这意味着该'Detergent_Paper'特征与其他预测变量高度相关，并且可以使用它们的信息来预测它。

# 可视化特征分布
# 对于数据中的每一对特征构造一个散布矩阵
# pd.scatter_matrix(data, alpha=0.3, figsize=(14, 8), diagonal='kde');

# 问题 3
# 这些功能的数据似乎具有线性关系。数据似乎不是正态分布的。
# 通过观察Detergents_Paper和Grocery、Milk具有一定程度的相关性。这证实了我的猜测。特征正偏斜。

# 数据预处理
# 练习: 特征缩放
# 使用自然对数缩放数据
log_data = np.log(data)

# 使用自然对数缩放样本数据
log_samples = np.log(samples)

# 为每一对新产生的特征制作一个散射矩阵
pd.scatter_matrix(log_data, alpha=0.3, figsize=(14, 8), diagonal='kde');

# 展示经过对数变换后的样本数据
# display(log_samples)

# 练习: 异常值检测
# 对于每一个特征，找到值异常高或者是异常低的数据点
for feature in log_data.keys():
    # 计算给定特征的Q1（数据的25th分位点）
    Q1 = np.percentile(log_data[feature], 25)

    # 计算给定特征的Q3（数据的75th分位点）
    Q3 = np.percentile(log_data[feature], 75)

    # 使用四分位范围计算异常阶（1.5倍的四分位距）
    step = (Q3 - Q1) * 1.5

    # 显示异常点
    # print("Data points considered outliers for the feature '{}':".format(feature))
    # display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])

# 可选：选择你希望移除的数据点的索引
outliers = [65, 66, 75, 128, 154]  # 这几个值重复出现

# 如果选择了的话，移除异常点
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop=True)

# 问题 4
# 这些数据仅是某些单独商店的特性，并不适合评价所有商店，因此可以剔除。

# 特征转换
# 练习: 主成分分析（PCA）
# 通过在good data上使用PCA，将其转换成和当前特征数一样多的维度
pca = PCA(n_components=6)
pca.fit(good_data)

# 使用上面的PCA拟合将变换施加在log_samples上
pca_samples = pca.transform(log_samples)

# 生成PCA的结果图
pca_results = vs.pca_results(good_data, pca)

# 问题 5
# 第一PC中的方差为：0.4430
# 第一和第二PC中的方差是：0.4430 + 0.2638 = 0.7068
# 第一，第二和第三PC的方差为：0.4430 + 0.2638 + 0.1231 = 0.8299
# 前4个PC的方差为：0.4430 + 0.2638 + 0.1231 + 0.1012 = 0.9311
# PC1：负权重Detergents_Paper与的Milk和Grocery正相关。此维度能体现零售商品的客户。
# PC2：负权重Fresh与Frozen和Delicatessen正相关。此维度能体现餐饮的客户。
# PC3：正权重Delicatessen与负权重上Fresh负相关。此维度能体现熟食的客户。
# PC4：正权重Frozen与负权重上Delicatessen负相关。此维度能体现冷冻食品的客户。

# 观察
# 展示经过PCA转换的sample log-data
# 观察样本数据的前四个维度的数值。考虑这和你初始对样本点的解释是否一致。
# display(pd.DataFrame(np.round(pca_samples, 4), columns=pca_results.index.values))
# 0号样本，可能是一家便利店。维度1为负数，可以体现其为零售商品的特性
# 1号样本，可能是一家餐厅，维度2为负数，可以体现其为餐饮客户的特性
# 2号样本，可能是一家综合超市，维度1为负数（且比0号样本更小），维度2位负数，可以体现其为零售商品的特性

# 练习：降维
# 如果大部分的方差都能够通过两个或者是三个维度进行表示的话，降维之后的数据能够被可视化。
# 通过在good data上进行PCA，将其转换成两个维度
pca = PCA(n_components=2)
pca.fit(good_data)

# 使用上面训练的PCA将good data进行转换
reduced_data = pca.transform(good_data)

# 使用上面训练的PCA将log_samples进行转换
pca_samples = pca.transform(log_samples)

# 为降维后的数据创建一个DataFrame
reduced_data = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])

# 观察
# 展示经过两个维度的PCA转换之后的样本log-data
# 这里的结果与一个使用六个维度的PCA转换相比较时，前两维的数值是保持不变的。
# display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))

# 可视化一个双标图（Biplot）
# 双标图是一个散点图，每个数据点的位置由它所在主成分的分数确定。坐标系是主成分（这里是Dimension 1 和 Dimension 2）。
# Create a biplot
# vs.biplot(good_data, reduced_data, pca)

# 观察
# 第一主成分：Milk、Grocery、Detergents_Paper
# 第二主成分：Delicatessen、Frozen、Fresh

# 聚类

# 问题 6
# 我将使用高斯混合模型用于数据，因为它更像数据以数字方式组织的方式的密度问题。
# 当在散点图中观察数据时，很清楚没有指定的集群，因此使用将为我定义集群的模型将是首选。

# 练习: 创建聚类
components = [5, 4, 3, 2]
for component in components:
    # 在降维后的数据上使用你选择的聚类算法
    clusterer = BayesianGaussianMixture(n_components=component, covariance_type='tied').fit(reduced_data)

    # 预测每一个点的簇
    preds = clusterer.predict(reduced_data)

    # 找到聚类中心
    centers = clusterer.means_

    # 预测在每一个转换后的样本点的类
    sample_preds = clusterer.predict(pca_samples)

    # 计算选择的类别的平均轮廓系数（mean silhouette coefficient）
    score = np.round(silhouette_score(reduced_data, preds), 2)

    # print("The score for n_component={0} is {1}".format(comp, score))

# 问题 7
# n_component = 5是0.36
# n_component = 4是0.34
# n_component = 3是0.39
# n_component = 2是0.42 -最好成绩（也就是分为两类）

# 聚类可视化
# 从已有的实现中展示聚类的结果，循环结束后 component=2
# vs.cluster_results(reduced_data, preds, centers, pca_samples)

# 练习: 数据恢复
# 反向转换中心点
log_centers = pca.inverse_transform(centers)

# 对中心点做指数转换
true_centers = np.exp(log_centers)

# 显示真实的中心点(我们可以通过施加一个反向的转换恢复这个点所代表的用户的花费。)
# 随机模拟两个用户
segments = ['Segment {}'.format(i) for i in range(0, len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns=data.keys())
true_centers.index = segments
# display(true_centers)

# 问题 8
# Segment 0 可能是一家生鲜超市，生鲜在50%以上
# Segment 1 可能是一家杂货商店，杂货在75%以上

# 问题 9
# 显示预测结果
for i, pred in enumerate(sample_preds):
    print("Sample point", i, "predicted to be in Cluster", pred)

# 问题 10
# 此更改不会平等地影响所有客户。该公司可以在实施公司范围内的变更之前，对客户的特定功能部分使用A/B测试，并逐渐确定影响或在受控模拟中确定影响。

# 问题 11
# 读取包含聚类结果的数据
cluster_data = pd.read_csv("cluster.csv")
y = cluster_data['Region']
X = cluster_data.drop(['Region'], axis=1)

# 划分训练集测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=24)

clf = RandomForestClassifier(random_state=24)
clf.fit(X_train, y_train)
print("使用cluster特征的得分", clf.score(X_test, y_test))

# 移除cluster特征
X_train = X_train.copy()
X_train.drop(['cluster'], axis=1, inplace=True)
X_test = X_test.copy()
X_test.drop(['cluster'], axis=1, inplace=True)
clf.fit(X_train, y_train)
print("不使用cluster特征的得分", clf.score(X_test, y_test))

# 监督学习可以用于对基于他们的产品消费选择3或5天递送服务并用于对接收客户进行分类的当前顾客进行分类。
# 输出可以帮助公司预测客户需求，并且在过程的开始或早期用来建议3-5交货服务。

# 可视化内在的分布
# 根据‘Channel‘数据显示聚类的结果
vs.channel_results(reduced_data, outliers, pca_samples)

# 问题 12
print()

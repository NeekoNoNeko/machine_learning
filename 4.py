
import pandas as pd
import numpy as np

# 1. 创建数据集
data = {
    '序号': range(15),
    '年龄': ['青年', '青年', '青年', '青年', '青年', '中年', '中年', '中年', '中年', '中年', '老年', '老年', '老年', '老年', '老年'],
    '有工作': ['否', '否', '是', '是', '否', '否', '否', '是', '否', '否', '否', '否', '是', '是', '否'],
    '有房子': ['否', '否', '否', '是', '否', '否', '否', '是', '是', '是', '是', '是', '否', '否', '否'],
    '信用': ['一般', '好', '好', '一般', '一般', '一般', '好', '好', '非常好', '非常好', '非常好', '好', '好', '非常好', '一般'],
    '类别': ['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否']
}
df = pd.DataFrame(data)
df = df.set_index('序号')

# 2. 定义计算熵的函数
def calculate_entropy(series):
    """计算给定序列的熵"""
    # 获取序列中每个值的计数
    value_counts = series.value_counts()
    # 计算每个值的概率
    probabilities = value_counts / len(series)
    # 计算熵
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# 3. 定义计算信息增益的函数
def calculate_information_gain(dataframe, feature_name, target_name):
    """计算指定特征相对于目标变量的信息增益"""
    # 计算目标变量的总熵
    total_entropy = calculate_entropy(dataframe[target_name])

    # 计算特征的加权平均熵
    weighted_entropy = 0
    # 获取特征的唯一值
    feature_values = dataframe[feature_name].unique()
    for value in feature_values:
        # 获取该特征值对应的子集
        subset = dataframe[dataframe[feature_name] == value]
        # 计算子集的权重（即该特征值在数据集中出现的频率）
        weight = len(subset) / len(dataframe)
        # 将子集的熵加到加权平均熵中
        weighted_entropy += weight * calculate_entropy(subset[target_name])

    # 信息增益是总熵减去加权平均熵
    information_gain = total_entropy - weighted_entropy
    return information_gain

# 4. 计算每个特征的信息增益
features = ['年龄', '有工作', '有房子', '信用']
target = '类别'

info_gains = {}
for feature in features:
    info_gains[feature] = calculate_information_gain(df, feature, target)

# 5. 打印结果并找出最大信息增益的特征
print("各个特征的信息增益:")
for feature, gain in info_gains.items():
    print(f"  - {feature}: {gain:.4f}")

max_gain_feature = max(info_gains, key=info_gains.get)
print(f"\n信息增益最大的特征是: '{max_gain_feature}' (增益为: {info_gains[max_gain_feature]:.4f})")
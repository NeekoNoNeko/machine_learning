
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

# 1. 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# 2. 数据标准化
# PCA对特征的尺度敏感，因此在应用PCA之前进行标准化是一个好习惯
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 应用PCA进行降维
# 我们将数据从4维降到2维以便于可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. 打印解释的方差比例
print("每个主成分解释的方差比例:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"  - 主成分 {i+1}: {ratio:.4f} (即 {ratio*100:.2f}%)")
print(f"总解释方差: {sum(pca.explained_variance_ratio_):.4f} (即 {sum(pca.explained_variance_ratio_)*100:.2f}%)")

# 5. 创建降维后的DataFrame
pca_df = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
target_df = pd.DataFrame(data=y, columns=['target'])
final_df = pd.concat([pca_df, target_df], axis=1)

# 6. 可视化降维后的数据
plt.figure(figsize=(10, 7))
colors = ['r', 'g', 'b']
for target_val, color in zip(range(len(target_names)), colors):
    indices_to_keep = final_df['target'] == target_val
    plt.scatter(final_df.loc[indices_to_keep, 'Principal Component 1'],
                final_df.loc[indices_to_keep, 'Principal Component 2'],
                c=color,
                s=50,
                label=target_names[target_val])

plt.title('PCA of Iris Dataset (2 Components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid()
plt.show()
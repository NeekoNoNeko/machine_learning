
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
import pandas as pd

# 1. 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# 2. 应用K-Means聚类
# 鸢尾花有3个已知的类别，所以我们将n_clusters设置为3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# 3. 评估聚类效果
# 使用调整兰德系数（Adjusted Rand Index）来评估聚类效果
# ARI的取值范围为[-1, 1]，值越大意味着聚类结果与真实情况越吻合
ari = adjusted_rand_score(y, y_kmeans)
print(f"K-Means 聚类的调整兰德系数 (ARI): {ari:.4f}")

# 4. 使用PCA进行降维以便可视化
# 将4维数据降至2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 5. 可视化聚类结果和真实标签
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 绘制K-Means聚类结果
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', edgecolor='k', s=50)
axes[0].set_title(f'K-Means Clustering (ARI: {ari:.2f})')
axes[0].set_xlabel('Principal Component 1')
axes[0].set_ylabel('Principal Component 2')
axes[0].legend(handles=scatter1.legend_elements()[0], labels=['Cluster 0', 'Cluster 1', 'Cluster 2'])

# 绘制真实标签
scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
axes[1].set_title('True Labels')
axes[1].set_xlabel('Principal Component 1')
axes[1].set_ylabel('Principal Component 2')
axes[1].legend(handles=scatter2.legend_elements()[0], labels=list(iris.target_names))

# 添加中心点到K-Means图
centers = kmeans.cluster_centers_
centers_pca = pca.transform(centers)
axes[0].scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centroids')
axes[0].legend()

plt.suptitle('K-Means Clustering vs. True Labels for Iris Dataset')
plt.show()
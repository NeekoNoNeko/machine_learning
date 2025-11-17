
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. 加载数据集
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# 2. 划分训练集和测试集
# 将数据集划分为80%的训练集和20%的测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 打印数据集大小信息
print(f"总样本数: {len(X)}")
print(f"训练集样本数: {len(X_train)}")
print(f"测试集样本数: {len(X_test)}")
print(f"特征数量: {X.shape[1]}")

# 3. 创建并训练逻辑回归模型
# 使用默认参数创建逻辑回归模型
log_reg = LogisticRegression(max_iter=10000) # 增加max_iter以确保收敛

# 在训练集上训练模型
log_reg.fit(X_train, y_train)

# 4. 在测试集上进行预测
y_pred = log_reg.predict(X_test)

# 5. 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
print(f"\n逻辑回归模型的准确率: {accuracy:.4f}")

# 我们可以查看一些预测结果和真实标签的对比
print("\n部分测试集样本的预测结果与真实标签对比:")
for i in range(10):
    print(f"样本 {i}: 预测值 = {y_pred[i]}, 真实值 = {y_test[i]}")
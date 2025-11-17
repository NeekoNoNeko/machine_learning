'''
加载波士顿房价数据集，并分别利用最小二乘法和梯度下降法解决回归问题。
'''
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler

def load_boston_data():
    '''加载波士顿房价数据集'''
    boston = fetch_openml(name='boston', version=1, as_frame=False)
    return boston.data, boston.target

def main():
    '''主函数'''
    X, y = load_boston_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 最小二乘法 (使用 scikit-learn 的 LinearRegression)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_train)
    print("scikit-learn 最小二乘法求得的截距：", lin_reg.intercept_)
    print("scikit-learn 最小二乘法求得的系数：", lin_reg.coef_)

    # 梯度下降法 (使用 scikit-learn 的 SGDRegressor)
    sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.01, random_state=42)
    sgd_reg.fit(X_train_scaled, y_train)
    print("scikit-learn 梯度下降法求得的截距：", sgd_reg.intercept_)
    print("scikit-learn 梯度下降法求得的系数：", sgd_reg.coef_)

if __name__ == "__main__":
    main()
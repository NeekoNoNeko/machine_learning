
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB

# 1. 创建数据集
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}
df = pd.DataFrame(data)

print("原始数据集:")
print(df)

# 2. 数据预处理：将分类数据转换为数值数据
encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

print("\n编码后的数据集:")
print(df)

# 3. 准备训练数据和标签
X = df.drop('PlayTennis', axis=1)
y = df['PlayTennis']

# 4. 创建并训练朴素贝叶斯模型
model = CategoricalNB()
model.fit(X, y)

# 5. 准备要预测的样本
new_sample_data = {
    'Outlook': 'Sunny',
    'Temp': 'Mild',
    'Humidity': 'Normal',
    'Windy': 'Strong'
}

# 使用之前创建的编码器来转换新样本
encoded_sample = []
encoded_sample.append(encoders['Outlook'].transform(['Sunny'])[0])
encoded_sample.append(encoders['Temperature'].transform(['Mild'])[0])
encoded_sample.append(encoders['Humidity'].transform(['Normal'])[0])
encoded_sample.append(encoders['Wind'].transform(['Strong'])[0])

# 6. 进行预测
new_sample_df = pd.DataFrame([encoded_sample], columns=X.columns)
prediction_encoded = model.predict(new_sample_df)

# 7. 将预测结果解码回原始标签
prediction = encoders['PlayTennis'].inverse_transform(prediction_encoded)

print(f"\n要预测的样本: {new_sample_data}")
print(f"预测结果: 是否打球? {prediction[0]}")
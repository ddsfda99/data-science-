import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 加载数据
df = pd.read_csv('fraudulent.csv')

# 处理缺失值，这里我们使用每列的众数来填充缺失值
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# 划分特征和标签
X = df_imputed.drop('y', axis=1)
y = df_imputed['y']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 特征缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 训练不同的分类模型
models = {
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC()
}

# 存储每个模型的F1值
f1_scores = {}

for name, model in models.items():
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测测试集
    y_pred = model.predict(X_test)
    
    # 计算F1值
    f1 = f1_score(y_test, y_pred)
    f1_scores[name] = f1
    print(f"{name} F1 Score: {f1}")

# 比较F1值，找出最好的模型
best_model_name = max(f1_scores, key=f1_scores.get)
print(f"The best model is {best_model_name} with an F1 score of {f1_scores[best_model_name]}")

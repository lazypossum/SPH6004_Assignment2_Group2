import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

# 1. 加载和预处理数据
df = pd.read_excel('Timestep5_train.xlsx')
df = df.interpolate()
df = df.drop('race', axis=1)
df = df.drop('icu_cat', axis=1)
df['hosp_admittime'] = pd.to_datetime(df['hosp_admittime']).astype('int64') // 10**9
df['hosp_dischtime'] = pd.to_datetime(df['hosp_dischtime']).astype('int64') // 10**9
df['icu_intime'] = pd.to_datetime(df['icu_intime']).astype('int64') // 10**9
df['icu_outtime'] = pd.to_datetime(df['icu_outtime']).astype('int64') // 10**9
df['charttime'] = pd.to_datetime(df['charttime']).astype('int64') // 10**9
# 1. 加载和预处理数据
df_1 = pd.read_excel('Timestep5_test.xlsx')
df_1 = df_1.interpolate()
df_1 = df_1.drop('icu_cat', axis=1)
df_1 = df_1.drop('race', axis=1)
df_1['hosp_admittime'] = pd.to_datetime(df_1['hosp_admittime']).astype('int64') // 10**9
df_1['hosp_dischtime'] = pd.to_datetime(df_1['hosp_dischtime']).astype('int64') // 10**9
df_1['icu_intime'] = pd.to_datetime(df_1['icu_intime']).astype('int64') // 10**9
df_1['icu_outtime'] = pd.to_datetime(df_1['icu_outtime']).astype('int64') // 10**9
df_1['charttime'] = pd.to_datetime(df_1['charttime']).astype('int64') // 10**9

df_2 = pd.read_excel('Timestep5_holdout.xlsx')
df_2 = df_2.interpolate()
df_2 = df_2.drop('icu_cat', axis=1)
df_2 = df_2.drop('race', axis=1)
df_2['hosp_admittime'] = pd.to_datetime(df_2['hosp_admittime']).astype('int64') // 10**9
df_2['hosp_dischtime'] = pd.to_datetime(df_2['hosp_dischtime']).astype('int64') // 10**9
df_2['icu_intime'] = pd.to_datetime(df_2['icu_intime']).astype('int64') // 10**9
df_2['icu_outtime'] = pd.to_datetime(df_2['icu_outtime']).astype('int64') // 10**9
df_2['charttime'] = pd.to_datetime(df_2['charttime']).astype('int64') // 10**9
# 2. 创建序列和划分数据集
def create_sequences(data, seq_length, target_column):
    xs, ys = [], []
    data_without_target = data.drop(columns=[target_column])  # 移除目标列以留下特征列
    target_column_values = data[target_column]  # 单独获取目标列的值

    for i in range(len(data) - seq_length - 1):
        x = data_without_target.iloc[i:(i + seq_length)].values  # 提取特征序列
        y = target_column_values.iloc[i + seq_length]  # 提取对应的目标值
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

seq_length = 5
X_train, y_train = create_sequences(df, seq_length, 'los_icu')
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, y_test = create_sequences(df_1, seq_length, 'los_icu')
X_holdout, y_holdout = create_sequences(df_2, seq_length, 'los_icu')
X_holdout = torch.tensor(X_holdout).to(dtype=torch.float32)
y_holdout = torch.tensor(y_holdout).to(dtype=torch.float32)

X_train = torch.tensor(X_train).to(dtype=torch.float32)
y_train = torch.tensor(y_train).to(dtype=torch.float32)
X_test = torch.tensor(X_test).to(dtype=torch.float32)
y_test = torch.tensor(y_test).to(dtype=torch.float32)
# 3. 定义RNN模型
class SimpleRNN(nn.Module):
    def __init__(self, hidden_layer_size=200):
        super(SimpleRNN, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.rnn = nn.RNN(67, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, 1)
        self.hidden_cell = torch.zeros(1, 1, self.hidden_layer_size)

    def forward(self, input_seq):
        rnn_out, self.hidden_cell = self.rnn(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(rnn_out.view(len(input_seq), -1))
        return predictions[-1]

# 4. 训练模型
model = SimpleRNN()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 30
best_val_loss = float("inf")
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for seq, labels in zip(X_train, y_train):
        optimizer.zero_grad()
        model.hidden_cell = torch.zeros(1, 1, model.hidden_layer_size)
        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()
        train_loss += single_loss.item()

    # 训练集损失
    train_loss = train_loss / len(X_train)

    # 在验证集上评估
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for seq, labels in zip(X_holdout, y_holdout):
            model.hidden_cell = torch.zeros(1, 1, model.hidden_layer_size)
            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            val_loss += single_loss.item()

    # 验证集损失
    val_loss = val_loss / len(X_holdout)

    # 打印训练和验证损失
    print(f'Epoch: {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    # 检查是否应该早停
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # 保存模型
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        # 如果验证损失在设定的epoch内没有改善，可以考虑停止训练
        print("Early stopping triggered.")
        break
model.eval()
with torch.no_grad():
    predictions = []
    for seq in X_test:
        model.hidden_cell = torch.zeros(1, 1, model.hidden_layer_size)
        predictions.append(model(seq).item())

# 计算测试集的RMSE
test_scores = np.array(predictions)
test_scores_tensor = torch.tensor(test_scores, dtype=torch.float32)

# 确保y_test也是同类型的Tensor
y_test = y_test.float()
test_rmse_tensor = torch.sqrt(torch.mean((test_scores_tensor - y_test)**2))

# 如果需要，将结果转换回numpy
#test_rmse = test_rmse_tensor.item()
print(f'Test RMSE: {test_rmse_tensor}')

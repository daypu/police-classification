import streamlit as st
import torch
import torch.nn as nn
import numpy as np

import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder

from transformers import BertTokenizer, BertConfig, BertModel, BertForSequenceClassification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 自定义BERT模型
class CustomBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size + 2, config.num_labels)  # 增加了2个特征

    def forward(self, input_ids=None, attention_mask=None, month=None, hour=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # BERT模型的pooler_output
        
        # 将时间特征拼接到pooled_output中
        time_features = torch.stack((month, hour), dim=1).float()  # 创建时间特征张量
        pooled_output = torch.cat((pooled_output, time_features), dim=1)  # 在最后一个维度上拼接
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        outputs = (logits,) + outputs[2:]  # 将 logits 与 BERT 模型的其他输出组合在一起
        
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        
        return outputs

# 加载BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 加载大类和小类编码器
major_encoder = joblib.load('major_encoder.pkl')
minor_encoder = joblib.load('minor_encoder.pkl')

# 初始化变量
num_major_labels = None
num_minor_labels = None
num_labels = None

# 打开并读取文件内容
with open('labels_info.txt', 'r') as f:
    lines = f.readlines()

# 解析文件内容并赋值给变量
for line in lines:
    if line.startswith("Number of major labels:"):
        num_major_labels = int(line.split(": ")[1].strip())
    elif line.startswith("Number of minor labels:"):
        num_minor_labels = int(line.split(": ")[1].strip())
    elif line.startswith("Total number of labels:"):
        num_labels = int(line.split(": ")[1].strip())

# 加载自定义模型权重
config = BertConfig.from_pretrained('bert-base-chinese', num_labels=num_labels)
model = CustomBertForSequenceClassification.from_pretrained('bert-base-chinese', config=config)
model.load_state_dict(torch.load('model.pth'))  # 加载训练好的模型权重
model.eval()

# 主函数
def main():
    st.title('警情分类系统')

    # 侧边栏设置日期和时间格式
    with st.sidebar:
        # date_format = st.selectbox("选择日期格式：", ['YYYY-MM-DD', 'MM/DD/YYYY', 'DD-MM-YYYY'])
        # time_format = st.selectbox("选择时间格式：", ['HH:MM:SS', 'HH:MM', 'HH'])

        # 用户选择上传CSV文件或手动输入数据
        st.write("### 选择数据输入方式：")
        option = st.radio("选择：", ('上传CSV文件', '手动输入数据'))


    if option == '上传CSV文件':
        # 上传 CSV 文件
        st.write("请确保csv表格包含列date, time, hour, content")
        use_preset_data = st.selectbox("使用预设数据：", ['不使用', 'data_cleaned.csv'])

        if use_preset_data != "不使用":
            df_preset = pd.read_csv(use_preset_data, encoding='utf-8')
            
            st.write(f"### 使用预设数据{use_preset_data}：")
            st.write(df_preset.head())

            if st.button("开始预测（预设数据）"):
                predict_and_display_results(df_preset)
        else:
            uploaded_file = st.file_uploader("上传CSV文件", type=['csv'])
            if uploaded_file is not None:
                # 读取上传的 CSV 文件
                df = pd.read_csv(uploaded_file, encoding='utf-8')
                
                # 预览上传文件的前5行数据
                st.write("### 数据预览：")
                st.write(df.head())
                
                # 执行预测并显示结果
                if st.button("开始预测"):
                    predict_and_display_results(df)
    
    elif option == '手动输入数据':
        num_samples = st.number_input("输入数据行数：", min_value=1, step=1, value=1)
        
        with st.expander("手动输入数据"):
            data = {
                'date': [],
                'time': [],
                'content': []
            }

            for i in range(num_samples):
                date = st.date_input(f"日期 {i+1}:")
                time = st.time_input(f"时间 {i+1}:")
                content = st.text_area(f"内容 {i+1}:")

                data['date'].append(date)
                data['time'].append(time)
                data['content'].append(content)

            df = pd.DataFrame(data)

            st.write("### 输入的数据预览：")
            st.write(df)

            # 执行预测并显示结果
            if st.button("开始预测"):
                st.write("### 推理结果：")
                predict_and_display_results(df)


# 执行预测并显示结果
def predict_and_display_results(df):
    input_ids, attention_mask, month, hour = preprocess_data(df)
    major_labels, minor_labels = predict(input_ids, attention_mask, month, hour)

    for i in range(len(df)):
        st.write(f"Sample {i+1}:")
        st.write(f"日期: {df['date'][i].strftime('%Y-%m-%d')}")
        st.write(f"时间: {df['time'][i].strftime('%H:%M:%S')}")
        st.write(f"内容: {df['content'][i]}")
        st.write(f"预测的大类标签: {major_labels[i]}")
        st.write(f"预测的小类标签: {minor_labels[i]}")
        st.write("---")

# 数据预处理函数
def preprocess_data(data):
    data['date'] = pd.to_datetime(data['date'])
    data['time'] = pd.to_datetime(data['time'], format='%H:%M:%S').dt.time
    data['month'] = data['date'].dt.month
    data['hour'] = data['time'].apply(lambda x: x.hour)

    tokenized = tokenizer(list(data['content']), padding='max_length', truncation=True, max_length=64, return_tensors='pt')

    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    month = torch.tensor(data['month'].values).to(device)
    hour = torch.tensor(data['hour'].values).to(device)
    
    return input_ids, attention_mask, month, hour

# 执行推理函数
def predict(input_ids, attention_mask, month, hour):
    total_samples = len(input_ids)
    major_labels_list = []
    minor_labels_list = []
    
    with torch.no_grad():
        for i in range(total_samples):
            outputs = model(input_ids=input_ids[i:i+1], attention_mask=attention_mask[i:i+1], month=month[i:i+1], hour=hour[i:i+1])
            logits = outputs[0]

            major_preds = np.argmax(logits[:, :num_major_labels].cpu().numpy(), axis=1)
            minor_preds = np.argmax(logits[:, num_major_labels:].cpu().numpy(), axis=1)

            major_labels = major_encoder.inverse_transform(major_preds)
            minor_labels = minor_encoder.inverse_transform(minor_preds)

            major_labels_list.append(major_labels[0])
            minor_labels_list.append(minor_labels[0])
    
    return major_labels_list, minor_labels_list

if __name__ == '__main__':
    main()

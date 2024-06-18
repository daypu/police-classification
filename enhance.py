import pandas as pd
from nlpcda import Similarword, RandomDeleteChar

# 读取原始的CSV文件
input_csv = 'your_input_file.csv'
output_csv = 'augmented_data.csv'

# 初始化增强方法
smw = Similarword(create_num=2, change_rate=0.5)
rdc = RandomDeleteChar(create_num=3, change_rate=0.3)

# 定义函数来对文本进行增强
def augment_text(text):
    augmented_texts = []
    
    # 同义词替换增强
    for _ in range(2):  # 创建两个增强文本
        augmented_text = smw.replace(text)[1]  # 取第二个增强后的文本
        augmented_texts.append(augmented_text)
    
    # 随机字删除增强
    for _ in range(3):  # 创建三个增强文本
        augmented_text = rdc.replace(text)[1]  # 取第二个增强后的文本
        augmented_texts.append(augmented_text)
    
    return augmented_texts

# 读取CSV文件
df = pd.read_csv(input_csv, sep='\t')  # 根据实际情况调整分隔符

# 对每行的 "content" 列进行增强，并保存增强后的数据
augmented_data = []

for index, row in df.iterrows():
    original_content = row['content']
    augmented_contents = augment_text(original_content)
    
    for augmented_content in augmented_contents:
        new_row = row.copy()
        new_row['content'] = augmented_content
        augmented_data.append(new_row)

# 将增强后的数据转换为DataFrame，并保存为新的CSV文件
augmented_df = pd.DataFrame(augmented_data)
augmented_df.to_csv(output_csv, sep='\t', index=False)  # 根据实际情况调整分隔符和其他参数

print(f"增强后的数据已保存至 {output_csv}")

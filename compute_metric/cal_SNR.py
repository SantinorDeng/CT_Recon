import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")

# 加载统计结果
mean_new1 = np.load('mean_new1.npy')
std_new1 = np.load('std_new1.npy')
mean_new2 = np.load('mean_new2.npy')
std_new2 = np.load('std_new2.npy')
mean_new3 = np.load('mean_new3.npy')
std_new3 = np.load('std_new3.npy')

# 定义 SNR 计算函数
def SNR(mean, std):
    return 10 * np.log10(mean / std)

# 计算 SNR
SNR_new1 = SNR(mean_new1, std_new1)
SNR_new2 = SNR(mean_new2, std_new2)
SNR_new3 = SNR(mean_new3, std_new3)

# 打印 SNR 结果
print("SNR_new1:", SNR_new1)
print("SNR_new2:", SNR_new2)
print("SNR_new3:", SNR_new3)

# 保存 SNR 结果
np.save('SNR_new1.npy', SNR_new1)
np.save('SNR_new2.npy', SNR_new2)
np.save('SNR_new3.npy', SNR_new3)

# 创建 DataFrame 并保存为 CSV
layer = np.array([
    [70, 120],
    [155, 205],
    [240, 290],
    [325, 375]
])
center = np.array([
    [210, 225],
    [400, 210],
    [470, 340],
    [350, 440],
    [200, 390]
])

df_SNR = pd.DataFrame({
    'New 1 SNR/dB': SNR_new1.flatten(),
    'New 2 SNR/dB': SNR_new2.flatten(),
    'New 3 SNR/dB': SNR_new3.flatten(),
    'ROI': np.tile([f'ROI_{i}' for i in range(len(center))], len(layer)),
    'Layer': np.repeat([f'Layer_{i}' for i in range(len(layer))], len(center))
})
df_SNR.to_csv("../table/SNR.csv")


# 绘制 SNR 柱状图
plt.figure(figsize=(10, 6))
sns.barplot(data=df_SNR.melt(id_vars=['ROI', 'Layer'], var_name='Data', value_name='SNR/dB'),
            x='Layer', y='SNR/dB', hue='Data')
plt.title('SNR Comparison')
plt.tight_layout()
plt.show()
plt.savefig("../fig/SNR_comparison.png")

# 绘制三个子图不同ROI的图表
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
df_new1 = df_SNR[['New 1 SNR/dB', 'ROI', 'Layer']].rename(columns={'New 1 SNR/dB': 'SNR/dB'})
sns.barplot(data=df_new1, x='Layer', y='SNR/dB', hue='ROI')
plt.title('New 1 SNR Comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.subplot(1, 3, 2)
df_new2 = df_SNR[['New 2 SNR/dB', 'ROI', 'Layer']].rename(columns={'New 2 SNR/dB': 'SNR/dB'})
sns.barplot(data=df_new2, x='Layer', y='SNR/dB', hue='ROI')
plt.title('New 2 SNR Comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 子图 3: new3 数据
plt.subplot(1, 3, 3)
df_new3 = df_SNR[['New 3 SNR/dB', 'ROI', 'Layer']].rename(columns={'New 3 SNR/dB': 'SNR/dB'})
sns.barplot(data=df_new3, x='Layer', y='SNR/dB', hue='ROI')
plt.title('New 3 SNR Comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
plt.savefig('../fig/SNR_ROI_comparison_3.png')
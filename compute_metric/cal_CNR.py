import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def CNR(mu1, mu2, sigma1, sigma2):
    return 10 * np.log10(np.abs(mu1 - mu2) / np.sqrt(sigma1**2 + sigma2**2))

def SNR(mean, std):
    return 10 * np.log10(mean / std)

# 加载数据
mean_new1 = np.load('mean_new1.npy')
std_new1 = np.load('std_new1.npy')
mean_new2 = np.load('mean_new2.npy')
std_new2 = np.load('std_new2.npy')
mean_new3 = np.load('mean_new3.npy')
std_new3 = np.load('std_new3.npy')

mean_bkgd_new1 = np.load('mean_bkgd_new1.npy')
std_bkgd_new1 = np.load('std_bkgd_new1.npy')
mean_bkgd_new2 = np.load('mean_bkgd_new2.npy')
std_bkgd_new2 = np.load('std_bkgd_new2.npy')
mean_bkgd_new3 = np.load('mean_bkgd_new3.npy')
std_bkgd_new3 = np.load('std_bkgd_new3.npy')

for i in range(5):
    for j in range(4):
        cnr = CNR(mean_new1[i, j], mean_bkgd_new1, std_new1[i, j], std_bkgd_new1)
        print(f"ROI {i + 1},Layer {j + 1} : CNR = {cnr} dB")

layer_num = 4
CNR_new1_list = []
CNR_new2_list = []
CNR_new3_list = []

# 计算 new1 的 CNR
for layer_idx in range(layer_num):
    snr_new1_layer = SNR(mean_new1[:, layer_idx], std_new1[:, layer_idx])
    max_snr_idx = np.argmax(snr_new1_layer)
    max_mean_new1 = mean_new1[max_snr_idx, layer_idx]
    max_std_new1 = std_new1[max_snr_idx, layer_idx]
    cnr_new1 = CNR(max_mean_new1, mean_bkgd_new1, max_std_new1, std_bkgd_new1)
    CNR_new1_list.append(cnr_new1)
    
print(CNR_new1_list)
# 计算 new2 的 CNR
for layer_idx in range(layer_num):
    snr_new2_layer = SNR(mean_new2[:, layer_idx], std_new2[:, layer_idx])
    max_snr_idx = np.argmax(snr_new2_layer)
    max_mean_new2 = mean_new2[max_snr_idx, layer_idx]
    max_std_new2 = std_new2[max_snr_idx, layer_idx]
    cnr_new2 = CNR(max_mean_new2, mean_bkgd_new2, max_std_new2, std_bkgd_new2)
    CNR_new2_list.append(cnr_new2)
print(CNR_new2_list)
# 计算 new3 的 CNR
for layer_idx in range(layer_num):
    snr_new3_layer = SNR(mean_new3[:, layer_idx], std_new3[:, layer_idx])
    max_snr_idx = np.argmax(snr_new3_layer)
    max_mean_new3 = mean_new3[max_snr_idx, layer_idx]
    max_std_new3 = std_new3[max_snr_idx, layer_idx]
    cnr_new3 = CNR(max_mean_new3, mean_bkgd_new3, max_std_new3, std_bkgd_new3)
    CNR_new3_list.append(cnr_new3)
print(CNR_new3_list)
# 创建 DataFrame
layer_labels = [f'layer{i + 1}' for i in range(layer_num)]
df = pd.DataFrame({
    'Layer': np.repeat(layer_labels, 3),
    'CNR/dB': CNR_new1_list + CNR_new2_list + CNR_new3_list,
    'Data': ['New 1'] * layer_num + ['New 2'] * layer_num + ['New 3'] * layer_num
})

# 保存为 CSV
df.to_csv("../table/CNR.csv", index=False)

# 绘制柱状图
plt.figure(figsize=(13, 5))
sns.barplot(data=df, x='Layer', y='CNR/dB', hue='Data')
plt.title('CNR Comparison for Each Layer')
plt.xlabel('Layer')
plt.ylabel('CNR/dB')
plt.legend(title='Data')
plt.show()
plt.savefig("../fig/CNR_Comparison.png")

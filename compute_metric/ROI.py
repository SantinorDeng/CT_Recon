import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk

recon_new1 = np.load('../recon_new1.npy')
recon_new2 = np.load('../recon_new2.npy')
recon_new3 = np.load('../recon_new3.npy')

# recon_new1 = (recon_new1 - recon_new1.min()) / (np.max(recon_new1) - np.min(recon_new1))
# recon_new2 = (recon_new2 - recon_new2.min()) / (np.max(recon_new2) - np.min(recon_new2))
# recon_new3 = (recon_new3 - recon_new3.min()) / (np.max(recon_new3) - np.min(recon_new3))

# 定义层面范围
layer = np.array([
    [70, 120],
    [155, 205],
    [240, 290],
    [325, 375]
])

# 定义圆盘半径
r = 55

# 定义 ROI 中心
center1 = np.array([
    [320,180], [470,250], [420,420], [220,420], [180,270]
])
center2 = np.array([
    [320,170], [470,240], [420,420], [220,420], [180,270]
])
center3 = np.array([
    [320,180], [470,250], [420,420], [220,420], [180,270]
])
def msk_generate(recon, center, layer, r):
    """
    生成 ROI 掩码
    :param recon: 重建后的 CT 图像数据
    :param center: ROI 中心坐标
    :param layer: 层面范围
    :param r: 圆盘半径
    :return: ROI 掩码
    """
    rr, cc = disk((center[0], center[1]), r, shape=recon.shape[1:])
    msk = np.zeros_like(recon, dtype=bool)
    msk[layer[0]:layer[1], rr, cc] = 1
    return msk

# 生成背景 ROI 掩码
ROI_bkgd = np.zeros(recon_new1.shape, dtype=bool)
ROI_bkgd[300:350, 600:630, 280:330] = 1
# plt.imshow(ROI_bkgd.sum(axis=(0)), cmap='gray')
# plt.title('Background ROI')
# plt.show()
# plt.savefig('../fig/background_ROI.png')
def generate_all_ROIs(recon, center, layer, r):
    ROI_list = []
    for center_idx in range(len(center)):
        for l in layer:
            ROI_list.append(msk_generate(recon, center[center_idx], l, r))
    return np.array(ROI_list).reshape(len(center), len(layer), *recon.shape)

# 生成所有 ROI 掩码
ROI_new1 = generate_all_ROIs(recon_new1, center1, layer, r)
ROI_new2 = generate_all_ROIs(recon_new2, center2, layer, r)
ROI_new3 = generate_all_ROIs(recon_new3, center3, layer, r)

print(ROI_new1.shape)

# 绘制 ROI 图像
# def plot_ROI(ROI, title):
#     plt.figure(figsize=(15, 5))
#     plt.subplot(1, 3, 1)
#     plt.imshow(ROI.sum(axis=(0, 1, 2)), cmap='gray', aspect='equal')
#     for center_idx in range(len(center1)):
#         x, y = center1[center_idx]
#         plt.text(y, x, str(center_idx + 1), color='red', ha='center', va='center', fontsize=12)
#     plt.title(f"{title} Transverse")
#     plt.subplot(1, 3, 2)
#     plt.imshow(ROI.sum(axis=(0, 1, 3)), cmap='gray', aspect='equal')
#     plt.title(f"{title} Axial")
#     plt.subplot(1, 3, 3)
#     plt.imshow(ROI.sum(axis=(0, 1, 4)), cmap='gray', aspect='equal')
#     plt.title(f"{title} Coronal")
#     plt.tight_layout() 
#     plt.show()
#     plt.savefig(f'../fig/{title}.png')

# plot_ROI(ROI_new1, 'ROI')

def calculate_stats(recon, ROI, ROI_bkgd, name):
    print(name)
    img_ROI = recon.copy()
    img_ROI = np.ma.asarray(img_ROI)
    std = np.zeros(ROI.shape[:2])
    mean = np.zeros(ROI.shape[:2])
    for i, j in np.ndindex(ROI.shape[:2]):
        img_ROI.mask = ~ROI[i, j]
        std[i, j] = img_ROI.std()
        mean[i, j] = img_ROI.mean()
        print(i, j, std[i, j], mean[i, j])
    img_ROI.mask = ~ROI_bkgd
    mean_bkgd = recon[ROI_bkgd].mean()
    std_bkgd = recon[ROI_bkgd].std()
    print(f"mean_bkgd_{name.lower()}:", mean_bkgd)
    print(f"std_bkgd_{name.lower()}:", std_bkgd)
    return mean, std, mean_bkgd, std_bkgd


mean_new1, std_new1, mean_bkgd_new1, std_bkgd_new1 = calculate_stats(recon_new1, ROI_new1, ROI_bkgd, 'New 1')
mean_new2, std_new2, mean_bkgd_new2, std_bkgd_new2 = calculate_stats(recon_new2, ROI_new2, ROI_bkgd, 'New 2')
mean_new3, std_new3, mean_bkgd_new3, std_bkgd_new3 = calculate_stats(recon_new3, ROI_new3, ROI_bkgd, 'New 3')

# 保存结果
np.save('mean_new1.npy', mean_new1)
np.save('std_new1.npy', std_new1)
np.save('mean_bkgd_new1.npy', mean_bkgd_new1)
np.save('std_bkgd_new1.npy', std_bkgd_new1)
np.save('mean_new2.npy', mean_new2)
np.save('std_new2.npy', std_new2)
np.save('mean_bkgd_new2.npy', mean_bkgd_new2)
np.save('std_bkgd_new2.npy', std_bkgd_new2)
np.save('mean_new3.npy', mean_new3)
np.save('std_new3.npy', std_new3)
np.save('mean_bkgd_new3.npy', mean_bkgd_new3)
np.save('std_bkgd_new3.npy', std_bkgd_new3)


# def plot_ROI_images(recon, ROI, layer, title):
#     img_ROI = np.zeros_like(recon)
#     for i, j in np.ndindex(ROI.shape[:2]):
#         img_ROI[ROI[i, j]] = recon[ROI[i, j]]

#     plt.figure(figsize=(20, 5))
#     for k in range(len(layer)):
#         plt.subplot(1, len(layer), k + 1)
#         plt.imshow(img_ROI[layer[k, 0]:layer[k, 1]].sum(axis=0),cmap='gray')
#         plt.title(f"{title} Layer {k + 1}")
#     plt.show()
#     plt.savefig(f'../fig/{title}.png')

# plot_ROI_images(recon_new1, ROI_new1, layer, 'ROI_on_Recon_New_1')
# plot_ROI_images(recon_new2, ROI_new2, layer, 'ROI_on_Recon_New_2')
# plot_ROI_images(recon_new3, ROI_new3, layer, 'ROI_on_Recon_New_3')

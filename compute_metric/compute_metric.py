import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.ndimage import center_of_mass
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ====================== 窗宽窗位预处理 ======================
def apply_ct_window(data, window_center, window_width):
    """CT值窗宽窗位预处理
    Args:
        data: 原始CT数据（HU值）
        window_center: 窗位（level）
        window_width: 窗宽（window）
    Returns:
        归一化后的数据[0-1]
    """
    min_hu = window_center - window_width / 2
    max_hu = window_center + window_width / 2
    windowed = np.clip(data, min_hu, max_hu)
    normalized = (windowed - min_hu) / (max_hu - min_hu)
    return normalized


# ====================== 聚类方法模块 ========================
def kmeans_cluster(flattened_data, n_clusters):
    """K-Means聚类"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(flattened_data)
    return {
        'labels': kmeans.labels_,
        'centroids': kmeans.cluster_centers_,
        'type': 'K-Means'
    }


def gmm_cluster(flattened_data, n_clusters):
    """高斯混合模型聚类"""
    gmm = GaussianMixture(n_components=n_clusters,
                          covariance_type='full',
                          random_state=0).fit(flattened_data)
    return {
        'labels': gmm.predict(flattened_data),
        'probabilities': gmm.predict_proba(flattened_data),
        'centroids': gmm.means_,
        'type': 'GMM'
    }


# ====================== 主处理函数 ========================
def calculate_metrics(npy_path, output_folder, Cluster,
                      window_center=40, window_width=400,
                      cluster_method='gmm', normalize=False):
    """改进后的主函数
    Args:
        window_center: 窗位（CT level）
        window_width: 窗宽（CT window）
        cluster_method: 聚类方法选择（'gmm'/'kmeans'）
    """
    # 读取数据
    data = np.load(npy_path)

    # 保存原始数据
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, Path(output_folder) / "data_orginal.nii.gz")

    # # 窗宽窗位预处理
    # if normalize:
    #     data = apply_ct_window(data, window_center, window_width)
    #     # 保存数据
    #     img = nib.Nifti1Image(data, np.eye(4))
    #     nib.save(img, Path(output_folder) / "data_normalized.nii.gz")

    # 计算特征层面
    layer_variances = np.var(data, axis=(1, 2))
    layer_means = np.mean(data, axis=(1, 2))
    combined_score = layer_means * layer_variances
    top_9_layers = np.argsort(combined_score)[::-1][:9]
    top_9_layers = [210, 220, 230, 240, 250, 260, 270, 280, 290]

    # 结果存储
    all_snr_results = []
    all_cnr_results = []

    for layer in top_9_layers:
        layer_data = data[layer, :, :]

        # 以图像中心为圆心，半径为250确定一个圆，获取圆内掩码
        center_y, center_x = layer_data.shape[0] // 2, layer_data.shape[1] // 2
        y, x = np.ogrid[:layer_data.shape[0], :layer_data.shape[1]]
        circle_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= 250 ** 2

        # 仅使用圆内的数据进行后续处理
        circle_indices = np.where(circle_mask)
        flattened = layer_data[circle_mask].reshape(-1, 1)

        # 选择聚类方法
        if cluster_method == 'gmm':
            cluster_result = gmm_cluster(flattened, Cluster)
        else:
            cluster_result = kmeans_cluster(flattened, Cluster)

        # 将聚类标签映射回原始图像的形状
        labels = np.zeros_like(layer_data)
        labels[circle_indices] = cluster_result['labels']

        # 计算每个聚类在图像空间中的质心坐标（y, x）
        centroids = []
        for i in range(Cluster):
            # 找到属于当前聚类的所有像素的坐标
            cluster_indices = np.where(labels == i)
            if len(cluster_indices[0]) > 0:  # 确保该聚类非空
                # 计算平均坐标作为质心
                centroid_y = np.mean(cluster_indices[0])
                centroid_x = np.mean(cluster_indices[1])
                centroids.append([centroid_y, centroid_x])

        # 确保我们有足够的质心
        if len(centroids) < Cluster:
            # 如果某些聚类为空，用随机位置填充
            for _ in range(Cluster - len(centroids)):
                centroids.append([
                    np.random.uniform(center_y - 250, center_y + 250),
                    np.random.uniform(center_x - 250, center_x + 250)
                ])

        radii = [5, 10, 15, 20, 25, 35, 45, 55]

        for radius in radii:
            # 以每个中心点为圆心生成一个小圆（2D）
            circle_masks = []
            for centroid in centroids:
                y, x = np.ogrid[:layer_data.shape[0], :layer_data.shape[1]]
                mask = (x - centroid[1]) ** 2 + (y - centroid[0]) ** 2 <= radius ** 2
                circle_masks.append(mask)

            # 计算每个小圆的SNR
            snr_values = []
            for mask in circle_masks:
                roi_mean = np.mean(layer_data[mask])
                roi_std = np.std(layer_data[mask])
                snr = roi_mean / roi_std
                snr_values.append(snr)

            # 处理可能的NaN值
            snr_values = [0 if np.isnan(snr) else snr for snr in snr_values]

            # 选择SNR最大的区域作为目标区域
            target_index = np.argmax(snr_values)
            target_roi = circle_masks[target_index]
            target_roi_mean = np.mean(layer_data[target_roi])
            target_roi_std = np.std(layer_data[target_roi])

            # 计算其他区域与目标区域的CNR值
            cnr_values = []
            for i in range(Cluster):
                if i != target_index:
                    background = circle_masks[i]
                    background_mean = np.mean(layer_data[background])
                    background_std = np.std(layer_data[background])
                    cnr = np.abs(target_roi_mean - background_mean) / np.sqrt(target_roi_std ** 2 + background_std ** 2)
                    cnr_values.append(cnr)

            all_snr_results.append(snr_values)
            all_cnr_results.append(cnr_values)

    # 保存每个区域的SNR值到CSV文件
    snr_df = pd.DataFrame(all_snr_results, columns=[f'ROI_{i}' for i in range(Cluster)],
                          index=[f'layer_{l}_radius_{r}' for l in top_9_layers for r in radii])
    snr_df.to_csv(Path(output_folder) / 'SNR_eachROI.csv')

    # 正确保存CNR值
    cnr_data = []
    for layer_idx, layer in enumerate(top_9_layers):
        for radius_idx, radius in enumerate(radii):
            snr_values = all_snr_results[layer_idx * len(radii) + radius_idx]
            target_index = np.argmax(snr_values)  # 最大SNR的亚区索引

            # 创建一个CNR结果的映射，默认值为NaN
            cnr_map = {i: np.nan for i in range(Cluster)}

            # 获取当前层和半径下的CNR值列表
            current_cnr_values = all_cnr_results[layer_idx * len(radii) + radius_idx]

            # 填充CNR值，跳过目标亚区
            cnr_idx = 0
            for i in range(Cluster):
                if i != target_index:
                    cnr_map[i] = current_cnr_values[cnr_idx]
                    cnr_idx += 1

            # 构建CNR行数据
            row = {
                '层数': layer,
                '半径': radius,
                '最大SNR亚区': target_index
            }

            # 添加每个亚区的CNR值
            for i in range(Cluster):
                row[f'CNR_{i}'] = cnr_map[i]

            cnr_data.append(row)

    # 创建并保存CNR DataFrame
    cnr_df = pd.DataFrame(cnr_data)
    cnr_df.to_csv(Path(output_folder) / 'CNR_by_radius.csv')

    # 保存选中的9副图像及聚类结果生成的小圆
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for idx, layer in enumerate(top_9_layers):
        layer_data = data[layer, :, :]
        clusters = labels  # 使用正确的聚类标签

        # 绘制原始图像
        axes[idx].imshow(layer_data, cmap='gray')
        axes[idx].set_title(f"Layer {layer} - Original Image")

        # 绘制聚类结果及小圆
        unique_clusters = np.unique(clusters)
        colors = ListedColormap(['red', 'green', 'blue', 'yellow', 'cyan', 'magenta'])
        for cluster in unique_clusters:
            cluster_mask = (clusters == cluster)
            axes[idx].imshow(cluster_mask, cmap=colors, alpha=0.5)

        for centroid in centroids:
            y, x = np.ogrid[:layer_data.shape[0], :layer_data.shape[1]]
            mask = (x - centroid[1]) ** 2 + (y - centroid[0]) ** 2 <= 100  # 假设半径为15
            axes[idx].imshow(mask, cmap='gray', alpha=0.5)

    plt.tight_layout()
    plt.savefig(Path(output_folder) / 'clustering_results.png')
    return None


# ====================== 使用示例 ========================
if __name__ == "__main__":
    # 三个数据的不同窗宽窗位设置
    configs = [
        {
            'npy_path': "/data1/wangfang_data/1_DoctorWorks/3_MedicalImaging/project/recon_new1.npy",
            'output_folder': "/data1/wangfang_data/1_DoctorWorks/3_MedicalImaging/project/recon_new1/",
            'window_center': 0.0533,
            'window_width': 0.1033,
            'cluster_method': 'gmm'
        },
        {
            'npy_path': "/data1/wangfang_data/1_DoctorWorks/3_MedicalImaging/project/recon_new2.npy",
            'output_folder': "/data1/wangfang_data/1_DoctorWorks/3_MedicalImaging/project/recon_new2/",
            'window_center': 16.23,
            'window_width': 32.09,
            'cluster_method': 'kmeans'
        },
        {
            'npy_path': "/data1/wangfang_data/1_DoctorWorks/3_MedicalImaging/project/recon_new3.npy",
            'output_folder': "/data1/wangfang_data/1_DoctorWorks/3_MedicalImaging/project/recon_new3/",
            'window_center': 5.54,
            'window_width': 11.09,
            'cluster_method': 'gmm'
        }
    ]

    for cfg in configs:
        calculate_metrics(
            npy_path=cfg['npy_path'],
            output_folder=cfg['output_folder'],
            Cluster=6,
            window_center=cfg['window_center'],
            window_width=cfg['window_width'],
            cluster_method=cfg['cluster_method']
        )
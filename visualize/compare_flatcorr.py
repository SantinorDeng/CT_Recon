import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import numpy as np
import matplotlib.pyplot as plt
from load_data import CT_data
import astra
def CT_recon(data:CT_data,init_angle=0,recon_algo="FDK_CUDA"):
    print('Start reconstruction...')
    print(data.data.shape)
    cone_geom = astra.create_proj_geom(
        'cone',
        1,
        1,
        data.data.shape[0],
        data.data.shape[2],
        np.linspace(init_angle, 2*np.pi+init_angle, data.img_num, endpoint=False),
        data.source_origin,
        0
    )

    sinogram_id = astra.data3d.create(datatype='-sino', data=data.data, geometry=cone_geom)
    vol_geom = astra.create_vol_geom(data.img_h,data.img_h,data.img_w)
    cfg = astra.astra_dict(recon_algo)
    cfg['ReconstructionDataId'] = astra.data3d.create(datatype='-vol', geometry=vol_geom,data=0)
    cfg['ProjectionDataId'] = sinogram_id
    alg_id = astra.algorithm.create(cfg)

    astra.algorithm.run(alg_id, 1)
    recon = astra.data3d.get(cfg['ReconstructionDataId'])
    sinogram = astra.data3d.get(cfg['ProjectionDataId'])
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(cfg['ReconstructionDataId'])
    astra.data3d.delete(cfg['ProjectionDataId'])
    print(recon.shape,sinogram.shape)
    print('Finish reconstruction...')
    return recon,sinogram

CT_data_new1 = CT_data('../../team12_new1')
CT_data_new1_noflat = CT_data('../../team12_new1',is_flat_correct=False)

recon_new1, sinogram_new1 = CT_recon(CT_data_new1)
recon_new1_noflat, sinogram_new1_noflat = CT_recon(CT_data_new1_noflat)

plt.figure(figsize=(10,5),constrained_layout=True)
plt.subplot(1,2,1)
plt.imshow(recon_new1[320],cmap='gray')
plt.xticks([])
plt.yticks([])
plt.xlabel("with flat correction")

plt.subplot(1,2,2)
plt.imshow(recon_new1_noflat[320],cmap='gray')
plt.xticks([])
plt.yticks([])
plt.xlabel("without flat correction")
plt.show()
plt.savefig("../fig/compare_flat_correction.png")
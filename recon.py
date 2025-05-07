import astra
from load_data import CT_data
import numpy as np
from numpy.fft import *
import os
from imageio import imwrite
def CT_recon(data:CT_data,init_angle=0,recon_algo="BP3D_CUDA"):
    print('Start reconstruction...')
    print(data.data.shape)
    ramp_window = np.ones(data.data.shape[2])
    ramp_window[:len(ramp_window)//2] = np.linspace(1,0,len(ramp_window)//2)
    ramp_window[len(ramp_window)//2:] = np.linspace(0,1,len(ramp_window)-len(ramp_window)//2)
    data_fourier = fftshift(fft(ifftshift(data.data,axes=2),axis=2),axes=2)
    data_fourier = data_fourier*ramp_window[np.newaxis,np.newaxis,:]
    data.data = (fftshift(ifft(ifftshift(data_fourier,axes=2),axis=2),axes=2))
    cone_geom = astra.create_proj_geom(
        # 'cone',
        # 1,
        # 1,
        # data.data.shape[0],
        # data.data.shape[2],
        # np.linspace(init_angle, 2*np.pi+init_angle, data.img_num, endpoint=False),
        # data.source_origin,
        # 0
        'parallel3d',
        1,
        1,
        data.data.shape[0],
        data.data.shape[2],
        np.linspace(init_angle, 2*np.pi+init_angle, data.img_num, endpoint=False),
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



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    outputdir_new1 = '../team12_new1_out'
    outputdir_new2 = '../team12_new2_out'
    outputdir_new3 = '../team12_new3_out'
    CT_data_new1 = CT_data('../team12_new1')
    CT_data_new2 = CT_data('../team12_new2')
    CT_data_new3 = CT_data('../team12_new3')

    recon_new1, sinogram_new1 = CT_recon(CT_data_new1)
    recon_new2, sinogram_new2 = CT_recon(CT_data_new2)
    recon_new3, sinogram_new3 = CT_recon(CT_data_new3)

    recon_new1[recon_new1 < 0] = 0
    recon_new1 /= np.max(recon_new1)
    recon_new1 = np.round(recon_new1 * 255).astype(np.uint8)
    
    recon_new2[recon_new2 < 0] = 0
    recon_new2 /= np.max(recon_new2)
    recon_new2 = np.round(recon_new2 * 255).astype(np.uint8)

    recon_new3[recon_new3 < 0] = 0
    recon_new3 /= np.max(recon_new3)
    recon_new3 = np.round(recon_new3 * 255).astype(np.uint8)

    # Show sinogram.
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(sinogram_new1[240], cmap='gray')
    plt.title("Sinogram of New1")
    plt.subplot(1, 3, 2)
    plt.imshow(sinogram_new2[240], cmap='gray')
    plt.title("Sinogram of New2")
    plt.subplot(1, 3, 3)
    plt.imshow(sinogram_new3[240], cmap='gray')
    plt.title("Sinogram of New3")
    plt.show()
    plt.savefig('./fig/sinogram.png')

    # Show reconstruction.
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(recon_new1[320], cmap='gray')
    plt.title("Recon of New1")
    plt.subplot(1, 3, 2)
    plt.imshow(recon_new2[320], cmap='gray')
    plt.title("Recon of New2")
    plt.subplot(1, 3, 3)
    plt.imshow(recon_new3[320], cmap='gray')
    plt.title("Recon of New3")
    plt.show()
    plt.savefig('./fig/reconstruction.png')

    # Save reconstruction.
    # if not os.path.isdir(outputdir_new1): 
    #     os.mkdir(outputdir_new1)
    # for i in range(300,341):
    #     im = recon_new1[i, :, :]
    #     im = np.flipud(im)
    #     imwrite(os.path.join(outputdir_new1, 'reco%03d.png' % i), im)
    
    # if not os.path.isdir(outputdir_new2): 
    #     os.mkdir(outputdir_new2)
    # for i in range(300,341):
    #     im = recon_new2[i, :, :]
    #     im = np.flipud(im)
    #     imwrite(os.path.join(outputdir_new2, 'reco%03d.png' % i), im)

    # if not os.path.isdir(outputdir_new3): 
    #     os.mkdir(outputdir_new3)
    # for i in range(300,341):
    #     im = recon_new3[i, :, :]
    #     im = np.flipud(im)
    #     imwrite(os.path.join(outputdir_new3, 'reco%03d.png' % i), im)
    
    # Save npy.
    np.save('recon_new1.npy',recon_new1)
    np.save('recon_new2.npy',recon_new2)
    np.save('recon_new3.npy',recon_new3)
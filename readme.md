# <center> CT Reconstruction </center>
## Introduction
This is a project for the course of Medical Imaging. The project is about the reconstruction of CT images. The reconstruction method is based on the Fourier Reconstruction and implemented in Python. Please notice that the reconstruction method is based on GPU-accelerated reconstruction with Astra package. **It means you should have a GPU with CUDA support to run the reconstruction method.**
## Installation
If you want to run the recon.py file, you need to install the following packages:
```bash
pip install numpy matplotlib scipy bs4 astra imageio
```
Hopefully, it will work. If not, please install other packages according to the errors.
## Usage
First of all, you should run the recon.py file to get three .npy files and the reconstructed images(saved in fig directory). You should change the input data directory and some CT_data class parameters in the recon.py file to get right results. 

Then, you can run the ROI.py file in compute_metric directory to get the ROI metrics, including mean of ROI, std of ROI, and corresponding background metrics.

Finally, you can run the cal_CNR.py and cal_SNR.py files in compute_metric directory to get the CNR and SNR of the reconstructed images.

## Things to note
The centers used by ROI.py are based on the ChatGPT estimation. It's fixed and not reliable. To be honest, it's really silly to use fixed centers and fuzzy radius to present a ROI circle. But if you ask me why I still use it, I will say my mom called me to have lunch so I don't have much time ^_^ .

You can address this flaw by implementing edge detection method or clustering method. These methods maybe can help you to get more accurate ROI.

Thank God. My partner in this CT_Recon project complete this task by using K-means clustering. You can directly run compute_metric.py in compute_metric directory to get the SNR and CNR.

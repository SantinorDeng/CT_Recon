import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter, median_filter
from bs4 import BeautifulSoup

class Calibration_data:
    def __init__(self, path) -> None:
        with open(path,encoding='utf-8') as f_calib:
            self.data_calib=BeautifulSoup(f_calib.read(), "xml")
        self.SourceToAxis=float(self.data_calib.find('SourceToAxis').text)
        self.AxisToDetector=float(self.data_calib.find('AxisToDetector').text)
        self.HorizLightSize=float(self.data_calib.find('HorizLightSize').text)
        self.VertLightSize=float(self.data_calib.find('VertLightSize').text)
        self.AxisOfRotationOffset=float(self.data_calib.find('AxisOfRotationOffset').text)
        self.EquatoralOffset=float(self.data_calib.find('EquatorialOffset').text)
        self.HorizPixelSize=float(self.data_calib.find('HorizPixelSize').text)
        self.VertPixelSize=float(self.data_calib.find('VertPixelSize').text)
class CT_data:
    def __init__(self,path,img_w=480,img_h=640,img_num=800,is_flat_correct=True,is_geo_correct=True) -> None:
        self.path_data = os.path.join(path,'ScanData')
        self.path_ref = os.path.join(path,'ScanRef')
        self.img_w, self.img_h, self.img_num = img_w, img_h, img_num
        self.scan_data = self.load_data(self.path_data)
        self.ref_data = self.load_data(self.path_ref)
        self.name = path.split('/')[-1].split('_')[-1]
        # Calib
        if is_flat_correct:
            self.data = np.log10(self.ref_data/self.scan_data)
        else:
            self.data = self.scan_data
        
        # filter
        # kernel_size = [4,1,3]
        # self.data = gaussian_filter(self.data, sigma=kernel_size)
        

        # Calibration
        self.Calibration = Calibration_data(os.path.join(path,'calibration.xml'))
        if is_geo_correct:
            self.data = np.roll(self.data, -int(self.Calibration.AxisOfRotationOffset), axis=2)
        else:
            pass
        # Calc Param
        self.calculate_param()

    def load_data(self,path,offset=27):
        data = np.zeros((self.img_w, self.img_num, self.img_h))
        for i in range(self.img_num):
            filename = f"{i:03d}.bmp"
            with open(os.path.join(path,filename), 'rb') as f:
                raw_data = np.fromfile(f, dtype=np.uint16)
                data_img = raw_data[offset:]
                img = data_img.reshape((self.img_w, self.img_h))  # Transpose to match MATLAB behavior
                data[:, i, :] = img
        return data

    def calculate_param(self):
        self.detector_pixel_size = self.Calibration.HorizPixelSize / self.img_h
        self.source_origin = (self.Calibration.SourceToAxis+self.Calibration.AxisToDetector)/self.detector_pixel_size

        # self.origin_det = self.Calibration.AxisToDetector/self.detector_pixel_size

if __name__ == "__main__":
    test_new1 = CT_data('../team12_new1')
    middle_slice = int(test_new1.img_w/2)
    plt.imshow(test_new1.data[middle_slice,:,:], cmap="gray", aspect="auto")
    plt.title("Sinogram")
    plt.xlabel("Projection Angle")
    plt.ylabel("Detector Position")
    plt.colorbar(label="Intensity")
    plt.show()
    plt.savefig("./fig/sinogram_test.png")

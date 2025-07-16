# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants as sconst
import torch
import imageio.v2 as imageio

# x,y,z coordinates of 16 antennas, customized for your own antenna array. 
# アンテナの位置 中心を0にしている
# ANT_LOC = [[-0.24, -0.24, 0], [-0.08, -0.24, 0], [0.08, -0.24, 0], [0.24, -0.24, 0],
#            [-0.24, -0.08, 0], [-0.08, -0.08, 0], [0.08, -0.08, 0], [0.24, -0.08, 0],
#            [-0.24,  0.08, 0], [-0.08,  0.08, 0], [0.08,  0.08, 0], [0.24,  0.08, 0],
#            [-0.24,  0.24, 0], [-0.08,  0.24, 0], [0.08,  0.24, 0], [0.24,  0.24, 0]]

ANT_LOC = [[-0.09, 0, 0], [0, 0, 0], [0.09, 0, 0]]

normalize = lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))

class Bartlett():
    """ Class to generate Spatial Spectrum using Bartlett Algorithm. """
    def __init__(self, frequency=920e6): #周波数=920MHz(本実験では．wifiで行う時はここを変える必要あり) 今回の実験ではどっち？？？
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.antenna_loc = torch.tensor(ANT_LOC, dtype=torch.float32).T  
        self.lamda = sconst.c / frequency #光速を周波数で割って波長を求めている
        self.theory_phase = self._calculate_theory_phase().to(self.device) #位相差の計算

    def _calculate_theory_phase(self):
        """ Calculates theoretical phase difference over both azimuthal and elevation angle. """
        azimuth = torch.linspace(0, 359, 360) / 180 * np.pi
        elevation = torch.linspace(1, 90, 90) / 180 * np.pi

        # azimuth[0,1,..0,1..], elevation [0,0,..1,1..]
        elevation_grid, azimuth_grid = torch.meshgrid(elevation, azimuth, indexing="ij")
        azimuth_grid = azimuth_grid.flatten()
        elevation_grid = elevation_grid.flatten()

        theory_dis_diff = (self.antenna_loc[0,:].unsqueeze(-1) * torch.cos(azimuth_grid) * torch.cos(elevation_grid) +
                        self.antenna_loc[1,:].unsqueeze(-1) * torch.sin(azimuth_grid) * torch.cos(elevation_grid))
        # x座標*cos(方位角)*cos(仰角) + y座標*sim(方位角)*cos(仰角) 各アンテナから信号までの距離を求めている
        theory_phase = -2 * np.pi * theory_dis_diff / self.lamda # 各アンテナが受信する信号がどれだけずれるか(＝位相差)
        return theory_phase.T

    def gen_spectrum(self, phase_measurements): # 式11(12)に対応している
        """ Generates spatial spectrum from phase measurements. """
        phase_measurements = torch.tensor(phase_measurements, dtype=torch.float32).to(self.device)
        delta_phase = self.theory_phase - phase_measurements.reshape(1, -1)   # (360x90,16) - 1x16
        phase_sum = torch.exp(1j * delta_phase).sum(1) / self.antenna_loc.shape[1] #　torch.exp(1j * delta_phase)が重みw
        spectrum = normalize(torch.abs(phase_sum)).view(90, 360).cpu().numpy()
        return spectrum


if __name__ == '__main__':
    
    sample_phase = [-1.886,-1.923,-2.832,-1.743,
                -1.751,-1.899,-2.370,-3.113,
                -2.394,-2.464,2.964,-2.904,
                -1.573,-2.525,-3.039,-2.839]

    
    worker = Bartlett()
    spectrum = worker.gen_spectrum(sample_phase)
    spectrum = (spectrum * 255).astype(np.uint8)

    imageio.imsave('spectrum.png', spectrum)
    #これで90×360の画像(白黒)が出来上がる

    #python gen_spectrum.pyで実行すると，sample_phaseについての画像が作成される．(画像はdataset_toolsフォルダに保存される)
    
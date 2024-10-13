# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 19:03:01 2024

@author: maxim
"""

import numpy as np
import matplotlib.pyplot as plt
import BD_functions_multilayer_sample as fm

tabepsmat = ['Glass','ITO','PEDOTPSS','PM66eC91_12','BCP','Ag','Air']

elayer = np.array([150, 30, 140, 5, 100], dtype=np.float64)

#活性层的位置
photoactive = 3

incident_angle = 0

lambmax = 1000.
lambmin = 300.
step_lamb = 10.

tab = fm.load_txt_files(np.arange(lambmin, lambmax+step_lamb*0.5, step_lamb), tabepsmat)

result_absorption = fm. main(tab, photoactive, elayer, lambmin, lambmax, step_lamb, incident_angle)

plt.figure(figsize=(6,5), tight_layout=True)
plt.plot(result_absorption[:,0], result_absorption[:,1], label="Absorption")
plt.ylim(0,100)
plt.legend()
plt.xlabel("Wavelength (nm)")
plt.show()

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

ds = Dataset('input.nc')
print(ds.variables.keys())

im = plt.imshow(ds.variables['thkobs'][:])
plt.colorbar(im)
plt.show()

im = plt.imshow(np.sqrt(ds.variables['uvelsurfobs'][:]**2 + ds.variables['vvelsurfobs'][:]**2))
plt.colorbar(im)
plt.show()

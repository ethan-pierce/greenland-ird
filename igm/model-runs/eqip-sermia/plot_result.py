import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

ds = Dataset('optimize.nc')

im = plt.imshow(ds.variables['thk'][-1])
plt.colorbar(im)
plt.show()

im = plt.imshow(ds.variables['velsurf_mag'][-1])
plt.colorbar(im)
plt.show()
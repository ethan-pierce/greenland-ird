import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

ds = Dataset('optimize.nc')
print(ds.variables.keys())

im = plt.imshow(ds.variables['thk'][-1])
plt.colorbar(im)
plt.title('Ice thickness (m)')
plt.show()

im = plt.imshow(ds.variables['velsurf_mag'][-1])
plt.colorbar(im)
plt.title('Surface velocity (m a$^{-1}$)')
plt.show()

im = plt.imshow(ds.variables['slidingco'][-1])
plt.colorbar(im)
plt.title('Sliding coefficient')
plt.show()

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

ds = Dataset('geology-optimized.nc')
print(ds.variables.keys())

im = plt.imshow(ds.variables['thk'][:])
plt.colorbar(im)
plt.title('Ice thickness (m)')
plt.show()

im = plt.imshow(ds.variables['velsurf_mag'][:])
plt.colorbar(im)
plt.title('Surface velocity (m a$^{-1}$)')
plt.show()

im = plt.imshow(np.sqrt(ds.variables['uvelbase'][:]**2 + ds.variables['vvelbase'][:]**2))
plt.colorbar(im)
plt.title('Sliding velocity (m a$^{-1}$)')
plt.show()

im = plt.imshow(ds.variables['slidingco'][:])
plt.colorbar(im)
plt.title('Sliding coefficient')
plt.show()

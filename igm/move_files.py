import os
import shutil

for f in os.listdir('./data/igm-inputs'):
    glacier = f.replace('.nc', '')
    
    try:
        os.mkdir('./igm/model-runs/' + glacier)
    except:
        pass
    
    shutil.copy('./data/igm-inputs/' + f, './igm/model-runs/' + glacier + '/observation.nc')

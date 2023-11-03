import os
import shutil

for f in os.listdir('./data/igm-inputs'):
    glacier = f.replace('.nc', '')
    
    try:
        os.mkdir('./igm/model-runs/' + glacier)
    except:
        pass
    
    try:
        os.remove('./igm/model-runs/' + glacier + '/input.nc')
    except:
        pass

    shutil.copy('./data/igm-inputs/' + f, './igm/model-runs/' + glacier + '/input.nc')

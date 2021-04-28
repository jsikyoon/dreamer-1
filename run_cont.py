import os, sys
import subprocess
from datetime import datetime

###############################################################################
# Params
###############################################################################
img_name = 'google-dreamer'

## gpu
gpu_ids = ['5']

###############################################################################
# Volumn options
###############################################################################
volumn_options = [
        "-v /common/home/jy651/:/jy651/ -v /data/local/jy651/:/data/local/jy651",
        "-v /cortex/users/jy651:/cortex/users/jy651"
        ]
volumn_options = " ".join(volumn_options) + " "

###############################################################################
# Run
###############################################################################

#cont_name = 'development'

command = 'docker run -it --rm '
#command = 'docker create -it '
command += volumn_options
command += \
    '--device=/dev/nvidiactl --device=/dev/nvidia-uvm --runtime nvidia '
for gpu_id in gpu_ids:
  command += '--device=/dev/nvidia'+gpu_id+' '
  command += '-e NVIDIA_VISIBLE_DEVICES='+gpu_id+' '
#command += '--name '+cont_name+' '
command += img_name
command += ' /bin/bash'
print(command)
os.system(command)

#os.system('docker start development')


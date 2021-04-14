import os, sys
import subprocess
from datetime import datetime

###############################################################################
# Params
###############################################################################
img_name = 'google-dreamer'

## gpu
gpu_ids = ['1']
cnt = 0
max_cnt = len(gpu_ids)

## tasks
tasks = ['dmlab_watermaze']
#tasks = ['dmlab_collect', 'dmlab_collect_few']

###############################################################################
# Volumn options
###############################################################################
volumn_options = [
        "-v /common:/common -v /data/local/jy651/:/data/local/jy651",
        "-v /cortex/users/jy651:/cortex/users/jy651"
        ]
volumn_options = " ".join(volumn_options) + " "

###############################################################################
# Run
###############################################################################

def run (task):

    global gpu_ids, cnt, max_cnt

    date = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    cont_name = '_'.join(['google-dreamer', task, date])

    run_command = "python3 -m dreamer.scripts.train --logdir ./logdir/"+task+" --params '{defaults: [dreamer, discrete], tasks: ["+task+"]}'"
    #run_command = "python3 while.py"

    command = 'docker run -d '
    #command += '--rm ' # when testing
    command += volumn_options
    command += \
        '--device=/dev/nvidiactl --device=/dev/nvidia-uvm --runtime nvidia '
    gpu_id = gpu_ids[cnt]
    cnt += 1
    cnt = cnt % max_cnt
    command += '--device=/dev/nvidia'+gpu_id+' '
    command += '-e NVIDIA_VISIBLE_DEVICES='+gpu_id+' '
    command += '--name '+cont_name+' '
    command += img_name
    command += ' /bin/bash -c "'
    #command += 'cd /common/home/jy651/gym-minigrid && pip install -e . && '
    command += 'cd /common/home/jy651/dreamer-1 && '
    command += run_command+'"'
    print(command)
    os.system(command)

for task in tasks:
    run(task)


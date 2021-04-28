python3 -m dreamer.scripts.train --logdir ./logdir/dmlab_collect_few \
    --params '{defaults: [dreamer, discrete, dmlab], tasks: [dmlab_collect_few]}'

python3 -m dreamer.scripts.train --logdir ./logdir/dmlab_collect_few/trxl \
    --params '{defaults: [dreamer, discrete, dmlab, trxl], tasks: [dmlab_collect_few]}'

python3 -m dreamer.scripts.train --logdir ./logdir/dmlab_collect_few/trxli \
    --params '{defaults: [dreamer, discrete, dmlab, trxli], tasks: [dmlab_collect_few]}'

python3 -m dreamer.scripts.train --logdir ./logdir/dmlab_collect_few/trxlgru \
    --params '{defaults: [dreamer, discrete, dmlab, trxl_gate_gru], tasks: [dmlab_collect_few]}'

#!/bin/bash
#SBATCH --job-name=mlp
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=08:00:00
#SBATCH --account=ds_ga_1006_001
#SBATCH --partition=n1c16m96-v100-2
#SBATCH --gres=gpu:v100:2


IMAGE=${IMAGE:-/scratch/amr1059/singularity/overlays/pytorch_21.06-py3.sif}
DATA_DIR=${DATA_DIR:-/scratch/amr1059/data}

singularity exec --no-home -B $HOME/.ssh -B /scratch -B $PWD --nv \
    --cleanenv \
    --overlay /scratch/amr1059/singularity/overlays/overlay-25GB-500K.ext3 \
    --overlay overlay-base.ext3:ro \
    --overlay overlay-packages.ext3:ro \
    --overlay $DATA_DIR/data.squashfs:ro \
    $IMAGE /bin/bash -c "
source /ext3/env.sh
conda activate /ext3/conda/bootcamp
cd capstone-project-47/
python -m mlp.train optim.learning_rate=5e-3 max_epochs=40
"

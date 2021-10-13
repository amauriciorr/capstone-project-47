#!/bin/bash
#SBATCH --job-name=mlp
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH -c 4

cd $SCRATCH
echo "running a job on node $(hostname) in directory $(pwd)"

singularity exec -B /scratch/ --nv \
    --overlay overlay-25GB-500K.ext3 \
    --overlay data/data.squashfs:ro \
    /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate
cd capstone-project-47/
python -m mlp.train batch_size=512 gpus=4 max_epochs=100
"

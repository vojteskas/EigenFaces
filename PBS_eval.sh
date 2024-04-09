#!/bin/bash
#PBS -N BZA_Eval
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:mem=100gb:scratch_ssd=100gb
#PBS -m ae

export OMP_NUM_THREADS=$PBS_NUM_PPN

name=BZA_Eval
archivename="$name"_Results.zip
DATADIR=/storage/brno2/home/vojteskas/deepfakes/eigenfaces


cd "$SCRATCHDIR" || exit 1
mkdir TMPDIR
export TMPDIR="$SCRATCHDIR/TMPDIR"


echo "Copying project files"
cp $DATADIR/* .
unzip Celeb-DF-v2-faces.zip >/dev/null 2>&1


echo "Creating conda environment"
module add gcc
module add conda-modules-py37
conda create --prefix "$TMPDIR/condaenv" python=3.10 -y >/dev/null 2>&1
conda activate "$TMPDIR/condaenv" >/dev/null 2>&1
pip install -r requirements.txt --cache-dir "$TMPDIR" >/dev/null 2>&1


chmod 755 ./*.py
echo "Running the script"
python eigenfaces.py


echo "Copying results"
zip -r "$archivename" ./*.png >/dev/null 2>&1
cp "$archivename" $DATADIR/$archivename >/dev/null 2>&1


clean_scratch

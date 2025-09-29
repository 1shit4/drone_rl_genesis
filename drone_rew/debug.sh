# expose debug info
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONFAULTHANDLER=1

# run training with headless rendering (to avoid OpenGL races)
export GENESIS_HEADLESS=1
python hover_train.py -e drone-hovering -B 16384 --max_iterations 999999

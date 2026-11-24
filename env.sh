# source this; all caches/tmp contained under /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study
export R=/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study
export XDG_CACHE_HOME=$R/cache/xdg PIP_CACHE_DIR=$R/cache/pip UV_CACHE_DIR=$R/cache/uv UV_PYTHON_INSTALL_DIR=$R/python
export TMPDIR=$R/tmp TORCH_EXTENSIONS_DIR=$R/cache/torch_extensions TORCHINDUCTOR_CACHE_DIR=$R/cache/inductor
export TRITON_CACHE_DIR=$R/cache/triton CUDA_CACHE_PATH=$R/cache/cuda TORCH_HOME=$R/cache/torch HF_HOME=$R/cache/hf
export MPLCONFIGDIR=$R/cache/mpl PYTHONNOUSERSITE=1 UV_NO_CONFIG=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES=0
# Contained CUDA 12.8.1 toolkit (extracted from NVIDIA runfile, matches torch cu128); used only for building FlashTP
export CUDA_HOME=$R/toolchain/cuda-12.8.1
export PATH=$CUDA_HOME/bin:$PATH

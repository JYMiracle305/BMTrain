# export NCCL_HOME=/home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/nccl-2.18.3-1-glfr3mzrcsf2ginrtjsgvmluxlcknko5
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_FILE=./nccl_debug_%h_%p.log
# export NCCL_P2P_DISABLE=0
# export CUDA_LAUNCH_BLOCKING=1
# LD_PRELOAD=$NCCL_HOME/lib/libnccl.so \
# python -m paddle.distributed.launch \
#        --gpus=0,1,2,3 \
#       ./train_by_paddle_multi.py \
#        --pipe_size 2 \
#        --tp_size 2
#export CUDA_VISIBLE_DEVICES=2
export NCCL_HOME=/home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/nccl-2.18.3-1-glfr3mzrcsf2ginrtjsgvmluxlcknko5
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=./nccl_debug_%h_%p.log
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
python -m paddle.distributed.launch \
       --gpus=2,3 \
      ./test_allreduce.py
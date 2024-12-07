set -x

export PYTHONPATH=$PWD:$PYTHONPATH

# Select the model type
# The model is downloaded to a specified location on disk, 
# or you can simply use the model's ID on Hugging Face, 
# which will then be downloaded to the default cache path on Hugging Face.

export MODEL_TYPE="CogVideoX"
# Configuration for different model types
# script, model_id, inference_step
declare -A MODEL_CONFIGS=(
    ["CogVideoX"]="parallel_sageattn_cogvideo.py THUDM/CogVideoX-2b 50"
)

if [[ -v MODEL_CONFIGS[$MODEL_TYPE] ]]; then
    IFS=' ' read -r SCRIPT MODEL_ID INFERENCE_STEP <<< "${MODEL_CONFIGS[$MODEL_TYPE]}"
    export SCRIPT MODEL_ID INFERENCE_STEP
else
    echo "Invalid MODEL_TYPE: $MODEL_TYPE"
    exit 1
fi

mkdir -p ./results

# task args
NUM_FRAMES=$1
if [ "$NUM_FRAMES" = "" ]; then
    NUM_FRAMES=49
fi

if [ "$MODEL_TYPE" = "CogVideoX" ]; then
  TASK_ARGS="--height 480 --width 720 --num_frames ${NUM_FRAMES} --max_sequence_length 226"
fi

# CogVideoX asserts sp_degree == ulysses_degree*ring_degree <= 2. Also, do not set the pipefusion degree.
if [ "$MODEL_TYPE" = "CogVideoX" ]; then
N_GPUS=2
# Only use CFG parallelism for 2 GPUs since it has minimal communication cost.
PARALLEL_ARGS="--ulysses_degree 1 --ring_degree 1"
CFG_ARGS="--use_cfg_parallel" 
fi

# COMPILE_FLAG=--use_torch_compile
# SAGE_ATTN_FLAG=--use_sage_attn_fp16
SAGE_ATTN_FLAG=--use_sage_attn_fp8
torchrun --nproc_per_node=$N_GPUS ./$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 0 \
$CFG_ARGS \
$PARALLLEL_VAE \
$COMPILE_FLAG \
$SAGE_ATTN_FLAG \
--prompt \
"A small dog."

# pretrained models
export MODEL_NAME="SG161222/Realistic_Vision_V4.0_noVAE"
export ENCODER_NAME="openai/clip-vit-large-patch14"

# our paths
export DV_PATH="models/disenvisioner/disvisioner.pt"
export EV_PATH="models/disenvisioner/envisioner.pt"

# inference settings
export IMAGE_PATH="assets/example_inputs/dog.jpg"
export SOBJ=0.7
export OUTDIR="output/dog/"

CUDA_VISIBLE_DEVICES=0 python run_disenvisioner.py \
    --pretrained_model_name_or_path $MODEL_NAME \
    --pretrained_CLIP $ENCODER_NAME \
    --output_dir $OUTDIR \
    --half_precision \
    --resolution 512 \
    --seed 42 \
    --num_samples 5 \
    --scale_object $SOBJ \
    --scale_others 0.0 \
    --disvisioner_path $DV_PATH \
    --envisioner_path $EV_PATH \
    --infer_image $IMAGE_PATH \
    --class_name "dog" \
    --infer_prompt "best quality, high quality, a * is running. "
# pretrained models
export MODEL_NAME="SG161222/Realistic_Vision_V4.0_noVAE"
export ENCODER_NAME="openai/clip-vit-large-patch14"
export IP_IMAGE_ENCODER_NAME="models/image_encoder"

# our paths
export DV_PATH="models/disenvisioner/disvisioner.pt"
export EV_PATH="models/disenvisioner/envisioner.pt"

# inference settings
export IMAGE_PATH="assets/example_inputs/candle.jpg"
export SOBJ=0.5
export SIP=0.4
export OUTDIR="output/candle/"


CUDA_VISIBLE_DEVICES=0 python run_disenvisioner_w_ip.py \
    --pretrained_model_name_or_path $MODEL_NAME \
    --pretrained_CLIP $ENCODER_NAME \
    --ip_image_encoder_path $IP_IMAGE_ENCODER_NAME \
    --output_dir $OUTDIR \
    --half_precision \
    --resolution 512 \
    --seed 42 \
    --num_samples 2 \
    --scale_object $SOBJ \
    --scale_others 0.0 \
    --scale_ip $SIP \
    --disvisioner_path $DV_PATH \
    --envisioner_path $EV_PATH \
    --infer_image $IMAGE_PATH \
    --class_name "candle" \
    --infer_prompt "best quality, high quality, a candle on a cobblestone street"
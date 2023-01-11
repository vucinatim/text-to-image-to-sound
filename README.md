## Running locally with PyTorch

_To successfully run the model, a GPU with at least 8gb vRAM is required.
For training at least 16gb vRAM._

### Installing the dependencies

```bash
git clone https://github.com/vucinatim/text-to-image-to-sound.git
cd text-to-image-to-sound
pip install -r requirements.txt
```

### Additional setup

If running on linux (we need to add some source library for a dependency)

```bash
apt-get update
apt install libsndfile1
```

Run the following command to authenticate your token to access the dataset on [HuggingFace](https://huggingface.co/datasets/vucinatim/spectrogram-captions) (this might not be needed)

```bash
huggingface-cli login
```

<br>

### Training

Initialize an [Accelerate](https://github.com/huggingface/accelerate/) environment with:
<sub>Every setting as default except the precission is recommended as fp16</sub>

```bash
accelerate config
```

Start the training with this command:

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export dataset_name="vucinatim/spectrogram-captions"

accelerate launch model/train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --use_ema \
  --resolution=256 \
  --train_batch_size=16 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --checkpointing_steps=1000 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="outputs"
```

Once the training is finished the model will be saved in the `output_dir` specified in the command. In this example it's `outputs`. 

### Running the trained model

To load the fine-tuned model for inference just pass that path to `StableDiffusionPipeline`

```python
from diffusers import StableDiffusionPipeline

model_path = "path_to_saved_model"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(prompt="birds chirping in the forest").images[0]
image.save("spectrogram.png")
```

## Running locally with PyTorch

### Installing the dependencies

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

Then cd in the example folder and run

```bash
pip install -r requirements.txt
```

And initialize an [Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

### Running example

Run the following command to authenticate your token

```bash
huggingface-cli login
```

<br>

#### Training

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export dataset_name="vucinatim/spectrogram-captions"

accelerate launch train_text_to_image.py \
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

Once the training is finished the model will be saved in the `output_dir` specified in the command. In this example it's `outputs`. To load the fine-tuned model for inference just pass that path to `StableDiffusionPipeline`

```python
from diffusers import StableDiffusionPipeline

model_path = "path_to_saved_model"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(prompt="birds chirping in the forest").images[0]
image.save("spectrogram.png")
```
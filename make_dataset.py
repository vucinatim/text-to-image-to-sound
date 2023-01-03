import librosa
import librosa.display
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.colors as colors
from tqdm import tqdm


# import cv2
from PIL import Image as PilImage
from datasets import Dataset, Image, Audio, DatasetInfo


def sound_to_img_hsv(audio_file_path, image_file_path):
    cmap = cm.get_cmap("hsv")
    norm = colors.Normalize(vmin=-80, vmax=0, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    y, sr = librosa.core.load(audio_file_path, sr=22050)
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256, hop_length=431)
    melspec = librosa.power_to_db(melspec, ref=np.max)

    image = np.zeros((melspec.shape[0], melspec.shape[1], 4), dtype=np.uint8)

    # Iterate over the elements of the 2D array and assign the corresponding color
    for i in range(melspec.shape[0]):
        for j in range(melspec.shape[1]):
            image[i, j, :] = mapper.to_rgba(melspec[i, j], bytes=True)

    im = PilImage.fromarray(image.astype(np.uint8))
    im.save(image_file_path)


def sound_to_img_bw(audio_file_path, image_file_path):
    y, sr = librosa.core.load(audio_file_path, sr=22050)
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256, hop_length=431)
    melspec = librosa.power_to_db(melspec, ref=np.max)

    image = (melspec + 80) / 80 * 255
    im = PilImage.fromarray(image.astype(np.uint8))
    im.save(image_file_path)


dataset_path = "./datasets/ERC-50"
audio_path = f"{dataset_path}/audio"
images_path = f"{dataset_path}/images"
prompts_path = f"{dataset_path}/prompts"
meta_path = f"{dataset_path}/meta"
metadata = pd.read_csv(f"{meta_path}/esc50.csv")

df = pd.DataFrame(columns=["id", "audio", "image", "text"])

total = 1000
for index, row in tqdm(metadata.head(total).iterrows(), total=total):

    audio_file_path = f"{audio_path}/{row['filename']}"
    image_file_path = f"{images_path}/bw/{row['filename'].replace('wav', 'png')}"

    # Generate spectrogram image from audio
    sound_to_img_bw(audio_file_path, image_file_path)
    # sound_to_img_hsv(audio_file_path, image_file_path)

    new_row = {
        "id": row["filename"].replace(".wav", ""),
        "audio": audio_file_path,
        "image": image_file_path,
        "text": f"{row['category'].replace('_', ' ')}",
    }

    # with open(f'{prompts_path}/{new_row["id"]}.txt', "w") as f:
    #     f.write(new_row["text"])

    df.loc[len(df.index)] = new_row

dataset = (
    Dataset.from_pandas(
        df,
        info=DatasetInfo(
            description="Dataset of captioned audio clips with spectrograms (text describing the sound).",
            citation="@misc{vucina2022spectrograms,\nauthor = {Vučina, Tim UNI-LJ},\ntitle = {Audio with spectrogram captions},\nyear={2022}}}",
        ),
        preserve_index=False,
    )
    .cast_column("image", Image())
    .cast_column("audio", Audio())
)

print(dataset[0])
dataset.push_to_hub("vucinatim/spectrogram-captions")

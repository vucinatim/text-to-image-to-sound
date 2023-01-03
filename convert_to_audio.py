from glob import glob
import librosa
import numpy as np
import soundfile as sf
from matplotlib import cm
import matplotlib.colors as colors
from tqdm import tqdm
from PIL import Image as PilImage


def get_value_from_cm(color, cmap, colrange):
    color = np.array(color) / 255.0
    r = np.linspace(colrange[0], colrange[1], 256)
    norm = colors.Normalize(colrange[0], colrange[1])
    mapvals = cmap(norm(r))[:, :4]
    distance = np.sum((mapvals - color) ** 2, axis=1)
    return r[np.argmin(distance)]


def img_hsv_to_sound(
    image_file, audio_file, sr=22050, hop_length=431, cmap=cm.get_cmap("hsv")
):
    image = PilImage.open(image_file)
    spec = np.zeros((image.height, image.width))
    for i in range(image.height):
        for j in range(image.width):
            spec[i][j] = get_value_from_cm(
                image.getpixel((j, i)), cmap, colrange=[-80, 0]
            )

    power_spec = librosa.db_to_power(spec)
    y = librosa.feature.inverse.mel_to_audio(power_spec, sr=sr, hop_length=hop_length)
    sf.write(audio_file, y, sr)


def img_bw_to_sound(image_file, audio_file, sr=22050, hop_length=431):
    image = PilImage.open(image_file)
    spec = np.asarray(image, dtype=np.float32) / 255 * 80 - 80
    power_spec = librosa.db_to_power(spec)
    y = librosa.feature.inverse.mel_to_audio(power_spec, sr=sr, hop_length=hop_length)
    sf.write(audio_file, y, sr)


images_path = "./outputs/images"
file_paths = glob(f"{images_path}/*.png")

for image_file_path in tqdm(file_paths, total=len(file_paths)):

    audio_file_path = image_file_path.replace("images", "audio").replace("png", "wav")

    # Generate audio from spectrogram image
    img_bw_to_sound(image_file_path, audio_file_path)
    # img_hsv_to_sound(image_file_path, audio_file_path)

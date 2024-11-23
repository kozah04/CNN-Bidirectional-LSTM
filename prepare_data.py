import os
from tqdm import tqdm
from configs import config
from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.tensorflow.dataProvider import DataProvider
from mltu.augmentors import RandomBrightness, RandomErodeDilate, RandomSharpen
from mltu.annotations.images import CVImage


def prepare_dataset(images_path, labels_path, configs):
    dataset = []
    vocab = set()
    max_len = 0

    with open(labels_path, "r") as file:
        sentences = file.readlines()

    for line in tqdm(sentences, desc="Loading Dataset"):
        if line.startswith("#") or 'err' in line:
            continue

        line_split = line.split(" ")
        folder1 = line_split[0][:3]
        folder2 = "-".join(line_split[0].split("-")[:2])
        file_name = line_split[0] + ".png"
        label = line_split[-1].strip().replace("|", " ")

        img_path = os.path.join(images_path, folder1, folder2, file_name)
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            continue

        dataset.append([img_path, label])
        vocab.update(list(label))
        max_len = max(max_len, len(label))

    configs.vocab = "".join(vocab)
    configs.max_text_length = max_len
    configs.save()

    data_provider = DataProvider(
        dataset=dataset,
        skip_validation=True,
        batch_size=configs.batch_size,
        data_preprocessors=[ImageReader(Image)],
        transformers=[
            ImageResizer(configs.width, configs.height, keep_aspect_ratio=True),
            LabelIndexer(configs.vocab),
            LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
        ],
    )

    train_data_provider, val_data_provider = data_provider.split(split=0.9)
    train_data_provider.augmentors = [RandomBrightness(), RandomErodeDilate(), RandomSharpen()]

    train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
    val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))

    return train_data_provider, val_data_provider

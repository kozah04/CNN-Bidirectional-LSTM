import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from mltu.tensorflow.callbacks import TrainLogger
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.metrics import CERMetric, WERMetric
from config import configs
from data_preparation import prepare_dataset
from model import train_model

images_path = '/kaggle/input/iam-sentence-dataset/IAM-SENTENCES-DATASET/sentences'
labels_path = '/kaggle/input/iam-sentence-dataset/IAM-SENTENCES-DATASET/metadata/sentences.txt'

train_data_provider, val_data_provider = prepare_dataset(images_path, labels_path, configs)
model = train_model((configs.height, configs.width, 3), len(configs.vocab))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate),
    loss=CTCloss(),
    metrics=[CERMetric(vocabulary=configs.vocab), WERMetric(vocabulary=configs.vocab)],
)

earlystopper = EarlyStopping(monitor="val_CER", patience=20, verbose=1, mode="min")
checkpoint = ModelCheckpoint(f"{configs.model_path}/model.keras", monitor="val_CER", save_best_only=True, mode="min", verbose=1)
train_logger = TrainLogger(configs.model_path)
tensorboard = TensorBoard(f"{configs.model_path}/training_logs")
reduce_lr = ReduceLROnPlateau(monitor="val_CER", factor=0.9, patience=5, verbose=1, mode="min")

model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=configs.train_epochs,
    callbacks=[earlystopper, checkpoint, train_logger, reduce_lr, tensorboard],
)

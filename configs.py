import os
from mltu.configs import BaseModelConfigs

class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join("CNN+Bidirectional-LSTM")
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.vocab = ""
        self.height = 96
        self.width = 1408
        self.max_text_length = 0
        self.batch_size = 32
        self.learning_rate = 0.0005
        self.train_epochs = 150

config = ModelConfigs()
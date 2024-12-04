# tsn_model.py
import torch.nn as nn
from torch.utils.data import Dataset

class TSNModel(nn.Module):
    def __init__(self, num_classes):
        super(TSNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# TSN 데이터셋 정의
class TSNDataset(Dataset):
    def __init__(self, train_list, classes_file, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        self.label_map = self.load_classes(classes_file)

        with open(train_list, "r") as f:
            for line_number, line in enumerate(f, start=1):
                try:
                    path, label_id = line.strip().rsplit(" ", 1)
                    if label_id not in self.label_map:
                        print(f"Unknown label ID '{label_id}' at line {line_number}, skipping...")
                        continue
                    label = int(label_id)
                    if os.path.exists(path):
                        self.data.append(path)
                        self.labels.append(label)
                    else:
                        print(f"Frame not found: {path}, skipping...")
                except ValueError as e:
                    print(f"Invalid format at line {line_number}: {line.strip()} - Error: {e}")

        if len(self.data) == 0:
            raise ValueError("No valid data found in train_list.txt")

class TSN:
    def __init__(self, model_weights_path):
        # TSN 모델 초기화
        self.model = self.load_model(model_weights_path)

    def load_model(self, model_weights_path):
        # 모델 로딩 코드 (예: PyTorch 사용)
        pass

    def train(self, image):
        # 모델 학습 코드
        pass

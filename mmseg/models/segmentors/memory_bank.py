import os
import torch
import time
from typing import Any, List, Optional
import torch.nn.functional as F

SAVE_DIR = "tools/ckpt_path"
os.makedirs(SAVE_DIR, exist_ok=True)

class DynamicMemory:
    def __init__(self):
        self.features: List[torch.Tensor] = []
        self.weights: List[str] = []

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        experiment_name = f"exp_{timestamp}"
        
        self.save_dir = os.path.join(SAVE_DIR, experiment_name)
        
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"save ckpt to: {self.save_dir}")

    def add(self, feature: torch.Tensor, weight: Any, domain_id: Optional[int] = None) -> None:
        feature = feature.float()
        self.features.append(feature)

        if domain_id is None:
            domain_id = len(self.features) - 1

        save_path = os.path.join(self.save_dir, f"domain_{domain_id}.pth")
        torch.save(weight, save_path)

        self.weights.append(save_path)

    def get_weight(self, index: int) -> str:
        if index < 0:
            index = len(self.weights) + index
        return self.weights[index]

    def compute_similarity(self, query):
        if not self.features:
            return None

        memory_features = torch.cat(self.features, dim=0)

        query = F.normalize(query.float().view(1, -1), dim=1)
        memory_features = F.normalize(memory_features, dim=1)

        return torch.matmul(query, memory_features.t()).squeeze()

    def ema_update(self, feature: torch.Tensor, index: int, alpha: float = 0.999):
        feature = feature.float()

        if index < 0:
            index = len(self.features) + index

        old_feature = self.features[index]
        new_feature = old_feature * alpha + feature * (1 - alpha)
        self.features[index] = new_feature 

    def update_weight(self, weight, index):
        path = self.weights[index]
        torch.save(weight, path)

    def __len__(self):
        return len(self.features)

    def clear(self) -> None:
        self.features.clear()
        self.weights.clear()

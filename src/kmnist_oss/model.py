import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.flatten = nn.Flatten()
        self.embedding = nn.Linear(64 * 7 * 7, 128)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)
        self.output = nn.Linear(128, 10)

    def forward_features(self, x):
        feature_map = self.features(x)
        flat_features = self.flatten(feature_map)
        embedding = self.embedding(flat_features)
        activated = self.activation(embedding)
        return {
            "feature_map": feature_map,
            "flat_features": flat_features,
            "embedding_pre_activation": embedding,
            "embedding": activated,
        }

    def forward_head(self, embedding, apply_dropout: bool = True):
        if apply_dropout:
            embedding = self.dropout(embedding)
        return self.output(embedding)

    def classifier(self, x):
        feature_dict = self.forward_features(x)
        return self.forward_head(feature_dict["embedding"])

    def forward(self, x, return_features: bool = False):
        feature_dict = self.forward_features(x)
        logits = self.forward_head(feature_dict["embedding"])
        if return_features:
            feature_dict["logits"] = logits
            return feature_dict
        return logits

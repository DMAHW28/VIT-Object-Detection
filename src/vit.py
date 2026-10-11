import torch
import torch.nn as nn
from collections import OrderedDict

class Patches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(B, -1, C, self.patch_size, self.patch_size)
        patches = torch.flatten(patches, start_dim = 2, end_dim = -1)
        return patches

class FeedForward(nn.Module):
    def __init__(self, hidden_dim, mlp_dim, dropout=0.25):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.feature_extractor(x)

class EncoderBlock(nn.Module):
    def __init__(
        self,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout,
        attention_dropout,
        ln_epsilon,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, eps=ln_epsilon)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=ln_epsilon)
        self.mlp = FeedForward(hidden_dim=hidden_dim, mlp_dim=mlp_dim, dropout=dropout)

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.self_attention(h, h, h, need_weights=False)
        h = self.dropout(h)
        h = h + x
        hh = self.norm2(h)
        hh = self.mlp(hh)
        return h + hh

class Encoder(nn.Module):
    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        ln_epsilon: float = 1e-6,
    ):
        super().__init__()
        self.position_encoding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        self.dropout = nn.Dropout(dropout)
        layers = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                ln_epsilon
            )
        self.layers = nn.Sequential(layers)
        self.norm = nn.LayerNorm(hidden_dim, eps=ln_epsilon)

    def forward(self, x):
        x = x + self.position_encoding
        return self.norm(self.layers(self.dropout(x)))

class ViT(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.image_size = params.get("image_size", 224)
        self.patch_size = params.get("patch_size", 16)
        self.hidden_dim = params.get("hidden_dim", 192)
        self.mlp_dim = params.get("mlp_dim", 768)
        self.attention_dropout = params.get("attention_dropout", 0.0)
        self.dropout = params.get("dropout", 0.0)
        self.num_classes = params.get("num_classes", 3)
        self.num_layers = params.get("num_layers", 12)
        self.num_heads = params.get("num_heads", 3)
        self.ln_epsilon = params.get("ln_epsilon", 1e-5)

        seq_length = (self.image_size // self.patch_size) ** 2
        self.conv_proj = nn.Conv2d(3, self.hidden_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        self.encoder = Encoder(
            seq_length+1,
            self.num_layers,
            self.num_heads,
            self.hidden_dim,
            self.mlp_dim,
            self.dropout,
            self.attention_dropout,
            self.ln_epsilon
        )

        self.cls_head = nn.Linear(self.hidden_dim, self.num_classes)
        self.box_head = nn.Linear(self.hidden_dim, 4)

    def forward(self, x):
        b, c, h, w = x.shape
        n_h = h // self.patch_size
        n_w = w // self.patch_size

        x = self.conv_proj(x)
        x = x.reshape(b, self.hidden_dim, n_h * n_w)
        x = x.permute(0, 2, 1)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = self.encoder(x)
        x = x[:, 0]
        cls_out = self.cls_head(x)
        box_out = self.box_head(x)
        return cls_out, box_out

class ViTP(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.image_size = params.get("image_size", 224)
        self.patch_size = params.get("patch_size", 16)
        self.hidden_dim = params.get("hidden_dim", 192)
        self.mlp_dim = params.get("mlp_dim", 768)
        self.attention_dropout = params.get("attention_dropout", 0.0)
        self.dropout = params.get("dropout", 0.0)
        self.num_classes = params.get("num_classes", 3)
        self.num_layers = params.get("num_layers", 12)
        self.num_heads = params.get("num_heads", 3)
        self.ln_epsilon = params.get("ln_epsilon", 1e-5)

        seq_length = (self.image_size // self.patch_size) ** 2
        self.patcher = Patches(self.patch_size)
        self.linear_projection = nn.Linear(3 * (self.patch_size ** 2), self.hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        self.encoder = Encoder(
            seq_length + 1,
            self.num_layers,
            self.num_heads,
            self.hidden_dim,
            self.mlp_dim,
            self.dropout,
            self.attention_dropout,
            self.ln_epsilon
        )

        self.cls_head = nn.Linear(self.hidden_dim, self.num_classes)
        self.boxes_head = nn.Linear(self.hidden_dim, 4)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.patcher(x)
        x = self.linear_projection(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = self.encoder(x)
        x = x[:, 0]
        cls_out = self.cls_head(x)
        box_out = self.boxes_head(x)
        return cls_out, box_out

if __name__ == "__main__":
    # --------------PATCHES---------------
    image_size = 224
    patch_size = 16

    # --------------TRANSFORMER ENCODER---------------
    hidden_dim = 192
    num_heads = 3
    num_layers = 12
    ln_epsilon = 1e-5
    mlp_dim = hidden_dim * 4
    dropout = 0.0
    attention_dropout = 0.0

    # --------------CLASSES---------------
    num_classes = 3

    params = {
        "patch_size": patch_size,
        "image_size": image_size,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "ln_epsilon": ln_epsilon,
        "mlp_dim": mlp_dim,
        "dropout": dropout,
        "attention_dropout": attention_dropout,
        "num_classes": num_classes,
    }

    model = ViT(params)
    print(model)
    x = torch.randn((1, 3, image_size, image_size))
    cls_out, box_out = model(x)
    print(box_out.shape, cls_out.shape)
    print(f"{sum([torch.numel(p) for p in model.parameters()]):,} parameters")

    model = ViTP(params)
    print(model)
    x = torch.randn((1, 3, image_size, image_size))
    cls_out, box_out = model(x)
    print(box_out.shape, cls_out.shape)
    print(f"{sum([torch.numel(p) for p in model.parameters()]):,} parameters")
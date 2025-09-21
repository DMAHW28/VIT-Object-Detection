import torch
import torch.nn as nn

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

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=None, activation="relu"):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = None
        self.dropout = None
        if dropout_rate is not None:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.25):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.feature_extractor(x)

class ClsHead(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.activation = params["cls_head_activation"]
        self.layers_units = params["cls_head_layers_units"]
        self.layers_dropout = params["cls_head_layers_dropout"]

        self.projection = nn.ModuleList()
        for i, (u, d) in enumerate(zip(self.layers_units, self.layers_dropout)):
            out_features = u
            in_features = self.layers_units[i - 1] if i > 0 else params["d_model"]
            activation = self.activation
            self.projection.append(LinearLayer(in_features=in_features, out_features=out_features, dropout_rate=d, activation=activation))

    def forward(self, x):
        for layer in self.projection:
            x = layer(x)
        return x

class TransformerEncoderLay(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.d_model = params["d_model"]
        self.num_heads = params["num_heads"]
        self.d_ff = params["d_ff"]
        self.ff_dropout = params["ff_dropout_rate"]
        self.epsilon = params["ln_epsilon"]
        self.norm1 = nn.LayerNorm(self.d_model, eps=self.epsilon)
        self.norm2 = nn.LayerNorm(self.d_model, eps=self.epsilon)
        self.attention_extractor = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=self.ff_dropout,
            batch_first=True,
        )
        self.ff = FeedForward(d_model=self.d_model, d_ff=self.d_ff, dropout=self.ff_dropout)

    def forward(self, x):
        x = self.norm1(x)
        attn_output, _ = self.attention_extractor( query = x, key = x, value = x)
        x = x + attn_output
        x = self.norm2(x)
        ff_output = self.ff(x)
        x = x + ff_output
        return x

class VitDetector(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.patcher = Patches(params["patch_size"])
        self.d_model = params["d_model"]
        self.n_layers = params["n_layers"]
        self.input_shape = params["input_shape"]
        self.n_patches = params["n_patches"]
        self.n_classes = params["n_classes"]

        # --------------LINEAR PROJECTION---------------
        self.linear_projection = nn.Linear(self.input_shape, self.d_model)

        # --------------EMBEDDING POSITION---------------
        self.register_parameter(name='cls_token', param=torch.nn.Parameter(torch.randn(1, 1, self.d_model)))
        self.position_encoding = nn.Embedding(self.n_patches + 1, self.d_model)

        # --------------TRANSFORMER---------------
        self.transformer = nn.Sequential(*[TransformerEncoderLay(params) for _ in range(self.n_layers)])

        # --------------PREDICTION---------------
        self.cls_head = nn.Linear(self.d_model, self.n_classes)
        self.boxes_head = nn.Linear(self.d_model, 4)

    def forward(self, x):
        B = x.size(0)
        x = self.patcher(x)
        x = self.linear_projection(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_emb = self.position_encoding(torch.arange(x.shape[1], device=x.device))
        x = x + pos_emb
        x = self.transformer(x)
        cls_tokens = x[:, 0]
        cls_out = self.cls_head(cls_tokens)
        box_out = self.boxes_head(cls_tokens)
        return cls_out, box_out

if __name__ == "__main__":
    # --------------PATCHES---------------
    img_size = 224
    patch_size = 16
    n_patches = (img_size * img_size) // (patch_size * patch_size)

    # --------------TRANSFORMER---------------
    input_shape = patch_size * patch_size * 3
    d_model = 192
    num_heads = 3
    n_layers = 12
    ln_epsilon = 1e-5
    d_ff = d_model * 4
    ff_dropout_rate = 0.25

    # --------------PREDICTION---------------
    num_classes = 3
    cls_head_activation = "gelu"
    cls_head_layers_units = [2048, 1024, 512, 64, 32]
    cls_head_layers_dropout = [0.25, 0.25, 0.25, 0.25, 0.25]

    params = {
        "patch_size": patch_size,
        "n_patches": n_patches,

        "input_shape": input_shape,
        "d_model": d_model,
        "num_heads": num_heads,
        "n_layers": n_layers,
        "ln_epsilon": ln_epsilon,
        "d_ff": d_ff,
        "ff_dropout_rate": ff_dropout_rate,

        "n_classes": num_classes,
        "cls_head_activation": cls_head_activation,
        "cls_head_layers_units": cls_head_layers_units,
        "cls_head_layers_dropout": cls_head_layers_dropout
    }

    model = VitDetector(params)
    x = torch.randn((1, 3, img_size, img_size))
    cls_out, box_out = model(x)
    print(box_out.shape, cls_out.shape)
    print(sum([torch.numel(para) for para in model.parameters()]))


"""
| Modèle                   | Dim. des embeddings | Têtes d’attention | Profondeur (couches) | Paramètres |
| ------------------------ | ------------------- | ----------------- | -------------------- | ---------- |
| **ViT-Tiny (ViT-Ti/16)** | 192                 | 3                 | 12                   | \~5.7M     |
| **ViT-Small (ViT-S/16)** | 384                 | 6                 | 12                   | \~22M      |
| **ViT-Base (ViT-B/16)**  | 768                 | 12                | 12                   | \~86M      |
| **ViT-Large (ViT-L/16)** | 1024                | 16                | 24                   | \~307M     |
| **ViT-Huge (ViT-H/14)**  | 1280                | 16                | 32                   | \~632M     |
"""

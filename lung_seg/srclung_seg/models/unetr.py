from monai.networks.nets import UNETR

from ..config import Config


def build_model(cfg: Config) -> UNETR:
    return UNETR(
        in_channels=1,
        out_channels=1,
        img_size=cfg.roi_size,
        feature_size=cfg.feature_size,
        hidden_size=cfg.hidden_size,
        mlp_dim=cfg.mlp_dim,
        num_heads=cfg.num_heads,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=cfg.dropout,
    )

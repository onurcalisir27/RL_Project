import hydra
from omegaconf import DictConfig

def load_config():
    @hydra.main(version_base="1.2", config_name="config", config_path="./cfg")
    def _load_config(cfg: DictConfig):
        config = {}
        for k, v in cfg.items():
            if isinstance(v, DictConfig):
                config[k] = dict(v)
            else:
                config[k] = v
        return config
    return _load_config()
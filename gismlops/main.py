import hydra
from gismlops.config import Params

# from gismlops.infer import infer
# from gismlops.train import train
from hydra.core.config_store import ConfigStore


cs = ConfigStore.instance()
cs.store(name="params", node=Params)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:
    print(cfg)
    # train()
    # infer()


if __name__ == "__main__":
    main()

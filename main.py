import hydra
from omegaconf import DictConfig
from models.reward_model import TrainWaypoint, EvaluateWaypoint
from models.sac_model import TrainSAC, EvaluateSAC

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(config: DictConfig):
    task = config.task.name
    if task == 'PickPlace':
        object_type = config.object
        if object_type not in ['bread', 'can', 'milk']:
            raise Exception('Unknown object type. \n Available objects: bread, can, milk')

    if config.method == 'waypoint':
        if config.train:
            TrainWaypoint(config)
        if config.test:
            EvaluateWaypoint(config)
    elif config.method == 'sac':
        if config.train:
            TrainSAC(config)
        if config.test:
            EvaluateSAC(config)
    else:
        raise ValueError("Invalid method specified. Use 'waypoint' or 'sac'.")

if __name__ == '__main__':
    main()

# This code is derived from here: https://recbole.io/docs/v1.0.0/user_guide/usage/use_modules.html

from __future__ import annotations

from logging import getLogger
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.context_aware_recommender.xdeepfm import xDeepFM
from recbole.model.general_recommender.bpr import BPR
from recbole.model.abstract_recommender import AbstractRecommender
from recbole.data.dataset.dataset import Dataset
from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function

def objective_function(config_dict=None, config_file_list=None):

    Model: str | AbstractRecommender = xDeepFM  # BPR
    dataset: str | Dataset = "example"

    config = Config(model=Model, dataset=dataset, config_dict=config_dict, config_file_list=config_file_list)

    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = Model(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    test_result = trainer.evaluate(test_data)

    logger.info(f'best valid result: {best_valid_result}')
    logger.info(f'test result: {test_result}')

    return {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }

if __name__ == '__main__':
    config_file_list: list[str]  = ["configs/basic.yaml", "configs/models/xdeepfm.yaml"]
    
    tuner = HyperTuning(
        objective_function=objective_function,
        algo="exhaustive",
        params_file="configs/hyperparams/xdeepfm.hyper",
        fixed_config_file_list=config_file_list,
    )

    tuner.run()


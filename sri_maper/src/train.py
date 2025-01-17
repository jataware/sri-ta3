from typing import List, Optional, Tuple

import hydra
from omegaconf import DictConfig
from torch import set_float32_matmul_precision
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import Logger
from sklearn.metrics import f1_score

from sri_maper.src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.
    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)
    
    log.info(f"Preprocessing rasters...")
    hydra.utils.call(cfg.preprocess)
    
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if "strategy" not in cfg.get("trainer") and model.net.contains_sync_batchnorm():
        # multi-GPU/CPU process train to single GPU/CPU process inference fix
        log.warning("Model checkpoint was trained with multi-GPU/CPU - reverting to single GPU/CPU")
        model.net.revert_sync_batchnorm()

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Testing best model from training!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        
        # preparation
        log.info(f"Best ckpt path: {ckpt_path}")
        model = model.__class__.load_from_checkpoint(ckpt_path)

        # temperature scaling
        if "temperature" not in cfg.model:
            model_calibrator = utils.BinaryTemperatureScaling(model)
            opt_temp = model_calibrator.calibrate(datamodule, cfg.trainer.limit_val_batches)
            del model_calibrator
            log.info(f"Optimal temperature: {opt_temp:.3f}")
            model.set_temperature(opt_temp)
        else:
            model.set_temperature(cfg.model.temperature)

        # theshold selection
        if "threshold" not in cfg.model:
            threshold_selector = utils.ThresholdMoving(model)
            opt_thr = threshold_selector.search_threshold(f1_score, datamodule, cfg.trainer.limit_val_batches)
            del threshold_selector
            log.info(f"Optimal threshold: {opt_thr:.3f}")
            model.set_threshold(opt_thr)
        else:
            model.set_threshold(cfg.model.threshold)

        log.info("Testing!")
        trainer.test(model=model, datamodule=datamodule)

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs/", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    set_float32_matmul_precision('medium')
    main()

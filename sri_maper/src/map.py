from typing import List, Optional, Tuple
import hydra
from omegaconf import DictConfig
from torch import set_float32_matmul_precision
from torch.distributed import get_rank
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
import pandas as pd

from sri_maper.src import utils

log = utils.get_pylogger(__name__)

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

@utils.task_wrapper
def build_map(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.
    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info(f"Preprocessing rasters...")
    hydra.utils.call(cfg.preprocess)
    
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, inference_mode=False)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    # preparation
    model = model.__class__.load_from_checkpoint(cfg.ckpt_path)
    
    if "strategy" not in cfg.get("trainer") and model.net.contains_sync_batchnorm():
        # multi-GPU/CPU process train to single GPU/CPU process inference fix
        log.warning("Model checkpoint was trained with multi-GPU/CPU - reverting to single GPU/CPU")
        model.net.revert_sync_batchnorm()

    if "temperature" not in cfg.model:
        datamodule.setup("validate")
        model_calibrator = utils.BinaryTemperatureScaling(model)
        opt_temp = model_calibrator.calibrate(datamodule, cfg.trainer.limit_val_batches)
        del model_calibrator
        log.info(f"Optimal temperature: {opt_temp:.3f}")
        model.set_temperature(opt_temp)
    else:
        model.set_temperature(cfg.model.temperature)
    
    model.hparams.extract_attributions = cfg.model.extract_attributions

    log.info("Starting map build!")
    trainer.predict(model=model, datamodule=datamodule)
    
    log.info(f"GPU:{trainer.strategy.global_rank} finished!")
    map_paths = None
    if trainer.strategy.global_rank == 0:
        log.info(f"GPU:{trainer.strategy.global_rank} is outputting map GeoTiff!")
        # read all GPU CSVs
        res_df = []
        for n in range(trainer.strategy.world_size):
            res_df.append(pd.read_csv(f"gpu_{n}_result.csv", index_col=False))
        res_df = pd.concat(res_df, ignore_index=True)
        
        tif_file_path = f"{cfg.paths.output_dir}"
        map_paths = utils.write_tif(res_df.values, tif_file_path, cfg.enable_attributions, datamodule)

    return map_paths, object_dict


@hydra.main(version_base="1.3", config_path="../configs/", config_name="test.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    utils.extras(cfg)
    # use the model to output a map
    build_map(cfg)


if __name__ == "__main__":
    set_float32_matmul_precision('medium')
    main()

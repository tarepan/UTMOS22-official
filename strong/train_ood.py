from collections import defaultdict
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from dataset import CVDataModule, TestDataModule
from lightning_module import BaselineLightningModule
import hydra
import wandb



@hydra.main(config_path="configs",config_name='default')
def train(cfg):
    debug = cfg.debug
    if debug:
        cfg.train.trainer_args.max_steps=10


    csvlogger = CSVLogger(save_dir=cfg.train.out_dir, name="train_log")
    tblogger = TensorBoardLogger(save_dir=cfg.train.out_dir, name="tf_log")
    wandblogger = WandbLogger(project="paper-train-ood-fixed",entity='sarulab-voicemos')

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.train.out_dir,
        save_weights_only=False,
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
        monitor="val_SRCC_system_ood",
        mode='max'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_callback,lr_monitor]
    earlystop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.0, patience=cfg.train.early_stopping.patience, mode="min"
    )
    callbacks.append(earlystop_callback)

    trainer = Trainer(
        **cfg.train.trainer_args,
        default_root_dir=hydra.utils.get_original_cwd(),
        limit_train_batches=0.01 if debug else 1.0,
        limit_val_batches=0.5 if debug else 1.0,
        callbacks=callbacks,
        logger=[csvlogger, tblogger, wandblogger],
    )

    datamodule = hydra.utils.instantiate(cfg.dataset.datamodule,cfg=cfg,_recursive_=False)
    test_datamodule = TestDataModule(cfg=cfg, i_cv=0, set_name='test')
    lightning_module = hydra.utils.instantiate(cfg.model.lightning_module,cfg=cfg , _recursive_=False)    
    wandblogger.watch(lightning_module)
    trainer.fit(lightning_module, datamodule=datamodule)

    if debug:
        trainer.test(lightning_module, datamodule=datamodule)
        trainer.test(lightning_module, datamodule=test_datamodule)
    else:
        trainer.test(lightning_module, datamodule=datamodule,ckpt_path=checkpoint_callback.best_model_path)
        trainer.test(lightning_module, datamodule=test_datamodule,ckpt_path=checkpoint_callback.best_model_path)
        wandb.save(checkpoint_callback.best_model_path)

if __name__ == "__main__":
    train()

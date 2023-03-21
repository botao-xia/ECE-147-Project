import argparse
import torch  
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.utilities.model_summary import summarize

from data_module import EEGDataModule
from model import LitModule

WANDB_USERNAME = "botao"

def main():
    parser = argparse.ArgumentParser()
    #IO related
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--ckpt_name', type=str, default="default_run")
    parser.add_argument('--ckpt_dir', type=str, default="./checkpoints")
    #dataloader related
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--do_data_prep', action='store_true')
    parser.add_argument('--random_state', type=int, default='42')
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--train_person_index', nargs='+', type=int, default=[]) # takes a list of subject indecies to train on
    parser.add_argument('--test_person_index', nargs='+', type=int, default=[]) # takes a list of subject indecies to test on
    parser.add_argument('--timestep', type=int, default=-1) # takes a list of subject indecies to test on
    parser.add_argument('--timestep_start', type=int, default=-1) # takes a list of subject indecies to test on
    parser.add_argument('--timestep_end', type=int, default=-1) # takes a list of subject indecies to test on
    #model runtime related
    parser.add_argument("--model_name", required=True, choices=['ShallowConvNet', 'ViTransformer', 'ATCNet', 'EEGNet_Modified'] ,help='model to use')
    parser.add_argument("--gpus", default='0', help='-1 means train on all gpus')
    parser.add_argument("--load_ckpt", default=None, type=str)
    parser.add_argument('--eval_only', action="store_true")
    parser.add_argument('--train_epochs', type=int, default=200)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_clip_val", default=1.0, type=float, help="Max gradient norm.")
    args = parser.parse_args()
    
    #seed everything
    seed_everything(args.random_state)

    #TODO: wandb logger; change to your own account to use it
    wb_logger = WandbLogger(project='EEG', name=args.ckpt_name, entity=WANDB_USERNAME)

    #data loaders
    dataModule = EEGDataModule(args)
    dataModule.setup()
    train_dataloader = dataModule.train_dataloader()

    #model
    trim = args.timestep_end - args.timestep_start
    model = LitModule(args.model_name, in_samples=(trim if trim != 0 else 1000))
    summary = summarize(model, max_depth=-1)
    print(summary)

    #trainer
    lr_logger = LearningRateMonitor() 
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        save_top_k=3,
        save_last=True,
        monitor='val_loss', # metric name 
        mode='min',
        save_weights_only=True,
        filename='{epoch}', # this cannot contain slashes
        )

    trainer = Trainer(
        accelerator="cpu", 
        devices=1,
        logger=wb_logger, 
        min_epochs=args.train_epochs,
        max_epochs=args.train_epochs, 
        gpus=str(args.gpus), # use string or list to specify gpu id  
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val, 
        num_sanity_val_steps=10,
        val_check_interval=1.0, # use float to check every n epochs 
        check_val_every_n_epoch=1,
        precision=32,
        auto_lr_find=True,
        callbacks = [lr_logger, checkpoint_callback]
    ) 

    if args.load_ckpt:
        checkpoint = torch.load(args.load_ckpt,map_location=model.device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)  
        model.on_load_checkpoint(checkpoint)

    elif args.eval_only: 
        trainer.test(model, datamodule=dataModule)
    else:
        #trainer.tune(model, datamodule=dataModule)
        trainer.fit(model, datamodule=dataModule) 
        trainer.test(model, datamodule=dataModule)
    

if __name__ == "__main__":
    main()









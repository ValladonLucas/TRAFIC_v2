import sys
sys.path.append("/work/luvallad/project")
import argparse
import pytorch_lightning as pl
import time
import copy

from Dataloaders.training_dataloader import DataModule
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback

from neptune import ANONYMOUS_API_TOKEN

from models.pointnet import PN
from models.pointnet_conf import PNConf
from models.dec import DEC
from models.dec_conf import DECConf
from models.seqdec import seqDEC
from models.seqdec_conf import seqDECConf

from utils.FuncUtils.utils import time_calculator, load_config

class DynamicModelCheckpoint(Callback):
    def __init__(self, checkpoint_callback, checkpoint_path):
        self.checkpoint_callback = checkpoint_callback
        self.checkpoint_path = checkpoint_path

    def on_train_start(self, trainer, pl_module):
        experiment_version = trainer.logger.version
        self.checkpoint_callback.dirpath = self.checkpoint_path
        self.checkpoint_callback.filename = f'{experiment_version}-{{epoch:02d}}-{{val_loss:.2f}}'
        trainer.callbacks.append(self.checkpoint_callback)

def main(args):
    
    # Initialize the model
    if args.model == "PNConf":
        model_type = PNConf
    elif args.model == "PN":
        model_type = PN
    elif args.model == "DEC":
        model_type = DEC
    elif args.model == "DECConf":
        model_type = DECConf
    elif args.model == "seqDEC":
        model_type = seqDEC
    elif args.model == "seqDECConf":
        model_type = seqDECConf
    else:
        print("Invalid model")
        return

    if args.mode == "full":
        print("Model: ", args.model)

        hparams = {"input_size": args.input_size, 
                    "n_classes": args.n_classes_1,
                    "embedding_size": args.embedding_size,
                    "k": args.k,
                    "dropout": args.dropout_1,
                    "aggr": args.aggr,
                    "num_heads": args.num_heads,
                    "noise": [args.noise_1, args.noise_range_1],
                    "rotation": [args.rotation_1, (-args.rotation_range_1, args.rotation_range_1)],
                    "shear": [args.shear_1, (-args.shear_range_1, args.shear_range_1)],
                    "translation": [args.translation_1, (-args.translation_range_1, args.translation_range_1)],
                    "probability": args.augmentation_prob_1,
                    }

        
        ############################################################################################################
        #                                               First model                                                #   
        ############################################################################################################

        # Initialize the data
        data_module_1 = DataModule(train_path=args.train_path_1,
                                 val_path=args.val_path_1,
                                 test_path=args.test_path_1,
                                 number_of_samples=args.n_samples_1,
                                 batch_size=args.batch_size_1,
                                 buffer_size=args.buffer_size_1,
                                 num_workers=args.num_workers)

        # Initialize the logger
        neptune_logger1 = NeptuneLogger(
            api_key=args.api_key,
            project=args.project,
            tags=[args.model, "first dataset", f"k={args.k}" if args.model in ["DEC", "DECConf", "seqDEC", "seqDECConf"] else ""],
        )
        neptune_logger1.log_hyperparams(hparams)

        # Initialize the early stopping
        early_stop_callback_1 = EarlyStopping(
            monitor=args.monitor,
            min_delta=args.min_delta,
            patience=args.patience,
            verbose=False,
            mode=args.early_stopping_mode
        )

        # Initialize the model checkpoint
        checkpoint_callback_1 = ModelCheckpoint(
            monitor=args.monitor,
            mode=args.early_stopping_mode,
            save_top_k=1,
            dirpath=None,
            filename=None
        )

        # Initialize the trainer
        trainer_1 = pl.Trainer(max_epochs=args.n_epochs, 
                            logger=neptune_logger1, 
                            log_every_n_steps=args.step_log_1, 
                            callbacks=[early_stop_callback_1, DynamicModelCheckpoint(checkpoint_callback_1, args.checkpoint_path_1)])

        # Train the model
        model1 = model_type(**hparams)
        start_time = time.time()
        trainer_1.fit(model1, data_module_1)
        h, m, s = time_calculator(start_time, time.time())
        print(f"First model training time: {h}h {m}m {s}s")

        # Load the best checkpoint
        best_model_path_1 = checkpoint_callback_1.best_model_path # After training finishes, use :attr:`best_model_path` to retrieve the path to the best checkpoint file
        best_model_1 = model_type.load_from_checkpoint(best_model_path_1)

        ############################################################################################################
        #                                               Second model                                               #
        ############################################################################################################

        # Initialize the data
        data_module_2 = DataModule(train_path=args.train_path_2,
                                val_path=args.val_path_2,
                                test_path=args.test_path_2,
                                number_of_samples=args.n_samples_2,
                                batch_size=args.batch_size_2,
                                buffer_size=args.buffer_size_2,
                                num_workers=args.num_workers)
        
        # Initialize the logger
        neptune_logger2 = NeptuneLogger(
            api_key=args.api_key,
            project=args.project,
            tags=[args.model, "second dataset", f"k={args.k}" if args.model in ["DEC", "DECConf", "seqDEC", "seqDECConf"] else ""],
        )

        # Initialize the early stopping
        early_stop_callback_2 = EarlyStopping(
            monitor=args.monitor,
            min_delta=args.min_delta,
            patience=args.patience,
            verbose=False,
            mode=args.early_stopping_mode
        )

        # Initialize the model checkpoint
        checkpoint_callback_2 = ModelCheckpoint(
            monitor=args.monitor,
            mode=args.early_stopping_mode,
            save_top_k=1,
            dirpath=None,
            filename=None
        )

        # Initialize the trainer
        trainer_2 = pl.Trainer(max_epochs=args.n_epochs, 
                            logger=neptune_logger2, 
                            log_every_n_steps=args.step_log_2, 
                            callbacks=[early_stop_callback_2, DynamicModelCheckpoint(checkpoint_callback_2, args.checkpoint_path_2)])

        # Load the best checkpoint and update hparams
        model2 = copy.deepcopy(best_model_1)
        new_hparams = {"input_size": args.input_size, 
                        "n_classes": args.n_classes_2,
                        "embedding_size": args.embedding_size,
                        "k": args.k,
                        "dropout": args.dropout_2,
                        "aggr": args.aggr,
                        "num_heads": args.num_heads,
                        "noise": [args.noise_2, args.noise_range_2],
                        "rotation": [args.rotation_2, (-args.rotation_range_2, args.rotation_range_2)],
                        "shear": [args.shear_2, (-args.shear_range_2, args.shear_range_2)],
                        "translation": [args.translation_2, (-args.translation_range_2, args.translation_range_2)],
                        "probability": args.augmentation_prob_2,
                        }
        model2.update_hparams(**new_hparams)
        
        # Train the model
        start_time = time.time()
        trainer_2.fit(model2, data_module_2)
        trainer_2.test(model2, data_module_2)
        h, m, s = time_calculator(start_time, time.time())
        print(f"Second model training time: {h}h {m}m {s}s")


    elif args.mode == "half":
        
        # Initialize the data
        data_module = DataModule(train_path=args.train_path,
                                 val_path=args.val_path,
                                 test_path=args.test_path,
                                 number_of_samples=args.n_samples,
                                 batch_size=args.batch_size,
                                 buffer_size=args.buffer_size,
                                 num_workers=args.num_workers)
        
        # Initialize the logger
        neptune_logger = NeptuneLogger(
            api_key=args.api_key,
            project=args.project,
            tags=[args.model, f"k={args.k}" if args.model in ["DEC", "DECConf", "seqDEC", "seqDECConf"] else ""],
        )

        # Initialize the early stopping
        early_stop_callback = EarlyStopping(
            monitor=args.monitor,
            min_delta=args.min_delta,
            patience=args.patience,
            verbose=False,
            mode=args.early_stopping_mode
        )

        # Initialize the model checkpoint
        checkpoint_callback = ModelCheckpoint(
            monitor=args.monitor,
            mode=args.early_stopping_mode,
            save_top_k=1,
            dirpath=None,
            filename=None
        )

        # Initialize the trainer
        trainer = pl.Trainer(max_epochs=args.n_epochs, 
                             logger=neptune_logger, 
                             log_every_n_steps=10, 
                             callbacks=[early_stop_callback, DynamicModelCheckpoint(checkpoint_callback, args.checkpoint_path)])

        new_hparams = {"input_size": args.input_size, 
                        "n_classes": args.n_classes,
                        "embedding_size": args.embedding_size,
                        "k": args.k,
                        "dropout": args.dropout,
                        "aggr": args.aggr,
                        "num_heads": args.num_heads,
                        "noise": [args.noise, args.noise_range],
                        "rotation": [args.rotation, (-args.rotation_range, args.rotation_range)],
                        "shear": [args.shear, (-args.shear_range, args.shear_range)],
                        "translation": [args.translation, (-args.translation_range, args.translation_range)],
                        "probability": args.augmentation_prob,
                        }
        print(args.pretrained_model)
        if args.pretrained_model is not None:
            print("\nOKOKOK\n")

            # Load the pretrained model
            pre_trained_model = model_type.load_from_checkpoint(args.pretrained_model)
        

            # Load the best checkpoint and update hparams
            model = copy.deepcopy(pre_trained_model)
            model.update_hparams(**new_hparams) # Unpack the dictionary and update the hparams
            neptune_logger.log_hyperparams(new_hparams) # Log the hyperparameters

        else:
            model = model_type(**new_hparams)
            neptune_logger.log_hyperparams(new_hparams)

        # Train the model
        start_time = time.time()
        trainer.fit(model, data_module)
        trainer.test(model, data_module)
        h, m, s = time_calculator(start_time, time.time())
        print(f"Training time: {h}h {m}m {s}s")

        

def get_argparse():
    parser = argparse.ArgumentParser(description='Trainer for models with low amount of data', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General
    parser.add_argument('--config', type=str, help='Path to the config file', default=None)

    # Mode
    parser.add_argument('--mode', type=str, help='Mode to run the script (full/half) if "full" pretrain + train if "half" provide pretrained model', default="full")

    namespace, _ = parser.parse_known_args()
    if namespace.mode == "full":
        # Models
        parser.add_argument('--model', type=str, help='Model to train')
        parser.add_argument('--n_epochs', type=int, help='Number of epochs')
        parser.add_argument('--num_workers', type=int, help='Number of workers', default=7)
        # First model data
        parser.add_argument('--train_path_1', type=str, help='Path to the training data of the first model')
        parser.add_argument('--val_path_1', type=str, help='Path to the validation data of the first model')
        parser.add_argument('--test_path_1', type=str, help='Path to the test data of the first model')
        parser.add_argument('--n_samples_1', type=int, help='Number of samples to use for the first model')
        parser.add_argument('--batch_size_1', type=int, help='Batch size of the first model')
        parser.add_argument('--buffer_size_1', type=int, help='Buffer size of the first model')
        parser.add_argument('--n_classes_1', type=int, help='Number of classes of the first model')
        parser.add_argument('--step_log_1', type=int, help='Step log of the first model', default=10)
        parser.add_argument('--checkpoint_path_1', type=str, help='Path to the checkpoint of the first model')
        parser.add_argument('--dropout_1', type=bool, help='Use dropout in the first model')
        parser.add_argument('--noise_1', type=bool, help='Use noise in the first model')
        parser.add_argument('--noise_range_1', type=float, help='Range of the noise in the first model')
        parser.add_argument('--rotation_1', type=bool, help='Use rotation in the first model')
        parser.add_argument('--rotation_range_1', type=float, help='Range of the rotation in the first model')
        parser.add_argument('--shear_1', type=bool, help='Use shear in the first model')
        parser.add_argument('--shear_range_1', type=float, help='Range of the shear in the first model')
        # Second model data
        parser.add_argument('--train_path_2', type=str, help='Path to the training data of the second model')
        parser.add_argument('--val_path_2', type=str, help='Path to the validation data of the second model')
        parser.add_argument('--test_path_2', type=str, help='Path to the test data of the second model')
        parser.add_argument('--n_samples_2', type=int, help='Number of samples to use for the second model')
        parser.add_argument('--batch_size_2', type=int, help='Batch size of the second model')
        parser.add_argument('--buffer_size_2', type=int, help='Buffer size of the second model')
        parser.add_argument('--n_classes_2', type=int, help='Number of classes of the second model')
        parser.add_argument('--step_log_2', type=int, help='Step log of the second model', default=10)
        parser.add_argument('--checkpoint_path_2', type=str, help='Path to the checkpoint of the second model')
        parser.add_argument('--dropout_2', type=bool, help='Use dropout in the second model')
        parser.add_argument('--noise_2', type=bool, help='Use noise in the second model')
        parser.add_argument('--noise_range_2', type=float, help='Range of the noise in the second model')
        parser.add_argument('--rotation_2', type=bool, help='Use rotation in the second model')
        parser.add_argument('--rotation_range_2', type=float, help='Range of the rotation in the second model')
        parser.add_argument('--shear_2', type=bool, help='Use shear in the second model')
        parser.add_argument('--shear_range_2', type=float, help='Range of the shear in the second model')
        # K
        parser.add_argument('--k', type=int, help='Number of nieghbors')

    elif namespace.mode == "half":
        # Models
        parser.add_argument('--pretrained_model', type=str, help='Path to the pretrained model')
        parser.add_argument('--model', type=str, help='Model to train')
        parser.add_argument('--n_epochs', type=int, help='Number of epochs')
        parser.add_argument('--num_workers', type=int, help='Number of workers', default=7)
        # Data
        parser.add_argument('--train_path', type=str, help='Path to the training data')
        parser.add_argument('--val_path', type=str, help='Path to the validation data')
        parser.add_argument('--test_path', type=str, help='Path to the test data')
        parser.add_argument('--n_samples', type=int, help='Number of samples to use')
        parser.add_argument('--n_classes', type=int, help='Number of classes')
        parser.add_argument('--batch_size', type=int, help='Batch size')
        parser.add_argument('--buffer_size', type=int, help='Buffer size')
        parser.add_argument('--step_log', type=int, help='Step log', default=10)
        parser.add_argument('--checkpoint_path', type=str, help='Path to the checkpoint')
        parser.add_argument('--dropout', type=bool, help='Use dropout in the new model')
        parser.add_argument('--noise', type=bool, help='Use noise in the new model')
        parser.add_argument('--noise_range', type=float, help='Range of the noise in the new model')
        parser.add_argument('--rotation', type=bool, help='Use rotation in the new model')
        parser.add_argument('--rotation_range', type=float, help='Range of the rotation in the new model')
        parser.add_argument('--shear', type=bool, help='Use shear in the new model')
        parser.add_argument('--shear_range', type=float, help='Range of the shear in the new model')
        # K
        parser.add_argument('--k', type=int, help='Number of nieghbors')

    # Neptune
    parser.add_argument('--api_key', type=str, help='Neptune API key', default=ANONYMOUS_API_TOKEN)
    parser.add_argument('--project', type=str, help='Neptune project')

    # Early stopping
    parser.add_argument('--monitor', type=str, help='Metric to monitor', default="val_loss")
    parser.add_argument('--min_delta', type=float, help='Minimum delta for early stopping', default=0.00)
    parser.add_argument('--patience', type=int, help='Patience for early stopping', default=10)
    parser.add_argument('--early_stopping_mode', type=str, help='Mode for early stopping', default="min")

    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        parser.set_defaults(**config)
        args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_argparse()
    main(args)
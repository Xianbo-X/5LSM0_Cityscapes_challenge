import torch
from torch import nn, optim
from torch.utils.data import DataLoader,ConcatDataset
from typing import Dict, Optional, Tuple, List
from scripts.dataset.dataset import CityscapesDataset
from scripts.metrics import compute_iou
import pandas as pd
import tqdm
import sys
import copy
from tqdm import tqdm

class Trainer:
    def __init__(self, model: nn.Module, ds_split: Dict[str,CityscapesDataset], learning_rate=0.001, writer=None,**kwargs):
        print(learning_rate)
        # Choose a device to run training on. Ideally, you have a GPU available to accelerate the training process.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Move the model onto the target device
        self.model = model.to(self.device)
        
        # Store the dataset split
        self.ds_split = ds_split

        # Tensorboard writer
        self.writer = writer
        
        ## EXERCISE #####################################################################
        #
        # Select an optimizer
        #
        # See: https://pytorch.org/docs/stable/optim.html
        #
        ################################################################################# 
        
        # Define Adam as the optimizer
        # reference: https://arxiv.org/pdf/2007.02839.pdf
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        
        ## EXERCISE #####################################################################
        #
        # Select an appropriate loss function
        #
        # See: https://pytorch.org/docs/stable/nn.html#loss-functions
        #
        ################################################################################# 
        
        # define Cross Entropy as the loss function
        # reference: https://arxiv.org/pdf/2006.14822.pdf
        self.critereon = nn.CrossEntropyLoss()
        
        ################################################################################# 
        
        assert self.critereon is not None, "You have not defined a loss"
        assert self.optimizer is not None, "You have not defined an optimizer"
        
    def train_epoch(self, dl:DataLoader, batch_size, epoch):
        # Put the model in training mode
        self.model.train()

        # Store the total loss and accuracy over the epoch
        amount = 0
        total_loss = 0
        total_accuracy = 0
        
        # Store each step's accuracy and loss for this epoch
        epoch_metrics = {
            "loss": [],
            "accuracy": []
        }
        
        # Create a progress bar using TQDM
        sys.stdout.flush()
        with tqdm(total=len(dl.dataset), desc=f'Training') as pbar:
            # Iterate over the training dataset
            for inputs, truths in dl:
                # Zero the gradients from the previous step
                self.optimizer.zero_grad()
                
                # Move the inputs and truths to the target device
                inputs = inputs.to(device=self.device, dtype=torch.float32)
                inputs.required_grad = True  # Fix for older PyTorch versions
                truths = truths.to(device=self.device, dtype=torch.long)
                
                # Run model on the inputs
                output = self.model(inputs)

                # Perform backpropagation
                loss = self.critereon(output, truths)
                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                
                # Store the metrics of this step
                step_metrics = {
                    'loss': loss.item(),
                    'accuracy': compute_iou(output, truths)
                }

                # Store loss and accuracy for visualization
                amount += 1
                total_loss += step_metrics["loss"]
                total_accuracy += step_metrics["accuracy"]

                if self.writer is not None:
                    self.writer.add_scalar("Loss/Minibatches/Training", step_metrics["loss"], amount+2975/batch_size*(epoch-1))
                    self.writer.add_scalar("Accuracy/Minibatches/Training", step_metrics["accuracy"], amount+2975/batch_size*(epoch-1))
                
                # Update the progress bar
                pbar.set_postfix(**step_metrics)
                pbar.update(list(inputs.shape)[0])
                
                # Add to epoch's metrics
                for k,v in step_metrics.items():
                    epoch_metrics[k].append(v)

        sys.stdout.flush()
        
        # loss and accuracy per epoch
        total_loss /= amount
        total_accuracy /= amount

        epoch_results = {
            "loss": [total_loss],
            "accuracy": [total_accuracy]
        }


        # Return metrics
        return epoch_metrics, epoch_results
    
    def val_epoch(self, dl:DataLoader, batch_size, epoch):
        # Put the model in evaluation mode
        self.model.eval()
        
        # Store the total loss and accuracy over the epoch
        amount = 0
        total_loss = 0
        total_accuracy = 0
        
        # Create a progress bar using TQDM
        sys.stdout.flush()
        with torch.no_grad(), tqdm(total=len(dl.dataset), desc=f'Validation') as pbar:
            # Iterate over the validation dataloader
            for inputs, truths in dl:
                 # Move the inputs and truths to the target device
                inputs = inputs.to(device=self.device, dtype=torch.float32)
                inputs.required_grad = True  # Fix for older PyTorch versions
                truths = truths.to(device=self.device, dtype=torch.long)

                # Run model on the inputs
                output = self.model(inputs)
                loss = self.critereon(output, truths)

                # Store the metrics of this step
                step_metrics = {
                    'loss': loss.item(),
                    'accuracy': compute_iou(output, truths)
                }

                # Update the progress bar
                pbar.set_postfix(**step_metrics)
                pbar.update(list(inputs.shape)[0])

                amount += 1
                total_loss += step_metrics["loss"]
                total_accuracy += step_metrics["accuracy"]

                # Store loss and accuracy for visualization
                if self.writer is not None:
                    self.writer.add_scalar("Loss/Minibatches/Validation", step_metrics["loss"], amount+500/batch_size*(epoch-1))
                    self.writer.add_scalar("Accuracy/Minibatches/Validation", step_metrics["accuracy"], amount+500/batch_size*(epoch-1))
        sys.stdout.flush()
        
        # Print mean of metrics
        total_loss /= amount
        total_accuracy /= amount
        print(f'Validation loss is {total_loss/amount}, validation accuracy is {total_accuracy}')
              
        # Return mean loss and accuracy
        return {
            "loss": [total_loss],
            "accuracy": [total_accuracy]
        }
            
        
    def fit(self, epochs: int, batch_size:int,aug_mode="None",start_epoch=1,result_foler=None,model_path_prefix=None,save_inter_model=True,**kwargs):
        """
        Parameters:
        ----------
        aug_mode: str
            "None": No augmentation during training
            "Only": Train on augmentation data only
            "Both" Train both origin data and augmentation data
        """
        # Initialize Dataloaders for the `train` and `val` splits of the dataset. 
        # A Dataloader loads a batch of samples from the each dataset split and concatenates these samples into a batch.
        train_set=self.ds_split["train"]
        val_set=self.ds_split["val"]
        if aug_mode=="Only":
            train_set.enable_aug()
            val_set.no_aug()
        if aug_mode=="None":
            train_set.no_aug()
            val_set.no_aug()
        if aug_mode=="Both":
            train_set.no_aug()
            val_set.no_aug()
            train_set_aug=copy.deepcopy(train_set)
            train_set_aug.enable_aug()
            train_set=ConcatDataset([train_set,train_set_aug])
        print(f"Aug mode={aug_mode}, daset_length={len(train_set)}")
        print(kwargs)
        print(f"result_foler: {result_foler}")
        print(f"model_prefix: {model_path_prefix}")

        dl_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        dl_val = DataLoader(val_set, batch_size=batch_size, drop_last=True)
                
        # Store metrics of the training process (plot this to gain insight)
        df_train = pd.DataFrame()
        df_val = pd.DataFrame()
        
        torch.save(self.model,model_path_prefix+f"_epoch_{0}.pt")
        # Train the model for the provided amount of epochs
        for epoch in range(start_epoch, start_epoch+epochs+1):
            print(f'Epoch {epoch}')
            metrics_train, train_epoch_results = self.train_epoch(dl_train, batch_size, epoch)
            if save_inter_model and (result_foler is not None and model_path_prefix is not None):
                torch.save(self.model,model_path_prefix+f"_epoch_{epoch}.pt")

            df_train = df_train.append(pd.DataFrame({'epoch': [epoch for _ in range(len(metrics_train["loss"]))], **metrics_train}), ignore_index=True)

            metrics_val = self.val_epoch(dl_val, batch_size, epoch)            
            df_val = df_val.append(pd.DataFrame({'epoch': [epoch], **metrics_val}), ignore_index=True) 

            # Store epoch results for visualization
            if self.writer is not None:
                self.writer.add_scalars("Loss/Epochs", 
                                {"Training Loss": train_epoch_results["loss"][0],
                                "Validation Loss": metrics_val["loss"][0]}, epoch)
                self.writer.add_scalars("Accuracy/Epochs", 
                                {"Training Accuracy": train_epoch_results["accuracy"][0],
                                "Validation Accuracy": metrics_val["accuracy"][0]}, epoch)
            
        # Return a dataframe that logs the training process. This can be exported to a CSV or plotted directly.
        return df_train, df_val

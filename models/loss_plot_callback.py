"""
Callback to plot training and validation losses vs epoch at the end of training.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger


class LossPlotCallback(Callback):
    """
    PyTorch Lightning callback to plot training and validation losses vs epoch.
    The plot is saved to the log directory at the end of training.
    """
    
    def __init__(self):
        super().__init__()
    
    def on_train_end(self, trainer: Trainer, pl_module):
        """
        Called when training ends. Plots training and validation losses vs epoch.
        """
        # Find CSV logger in the list of loggers
        csv_logger = None
        for logger in trainer.loggers:
            if isinstance(logger, CSVLogger):
                csv_logger = logger
                break
        
        if csv_logger is None:
            print("Warning: CSVLogger not found. Cannot plot losses.", flush=True)
            return
        
        # Get the metrics CSV file path
        metrics_file = os.path.join(csv_logger.log_dir, "metrics.csv")
        
        if not os.path.exists(metrics_file):
            print(f"Warning: Metrics file not found at {metrics_file}. Cannot plot losses.", flush=True)
            return
        
        try:
            # Read the metrics CSV file
            df = pd.read_csv(metrics_file)
            
            # Extract epochs and losses
            # Group by epoch and collect the last non-null train_loss and valid_loss for each epoch
            epochs = []
            train_losses = []
            valid_losses = []
            
            # Get unique epochs
            unique_epochs = sorted(df['epoch'].dropna().unique())
            
            for epoch in unique_epochs:
                epoch_data = df[df['epoch'] == epoch]
                
                # Get per-subject macro training loss (train_loss_macro) if available, otherwise fall back to train_loss
                train_loss_macro_vals = epoch_data['train_loss_macro'].dropna()
                
                # Get per-subject macro validation loss (valid_loss_macro) if available, otherwise fall back to valid_loss
                valid_loss_macro_vals = epoch_data['valid_loss_macro'].dropna()
                
                # Only add if we have both train and valid losses for this epoch
                if len(train_loss_macro_vals) > 0 and len(valid_loss_macro_vals) > 0:
                    # Take the last value (most recent) for each
                    epochs.append(int(epoch))
                    train_losses.append(float(train_loss_macro_vals.iloc[-1]))
                    valid_losses.append(float(valid_loss_macro_vals.iloc[-1]))
            
            if len(epochs) == 0:
                print("Warning: No valid loss data found in metrics.csv. Cannot plot losses.", flush=True)
                return
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(epochs, train_losses, 'b-', label='Training Loss (Per-Subject Macro)', linewidth=2, marker='o', markersize=4)
            ax.plot(epochs, valid_losses, 'r-', label='Validation Loss (Per-Subject Macro)', linewidth=2, marker='s', markersize=4)
            
            ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
            ax.set_title('Training and Validation Loss vs Epoch', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Set x-axis to show integer epochs
            ax.set_xticks(epochs)
            
            plt.tight_layout()
            
            # Save the plot to the log directory
            plot_path = os.path.join(csv_logger.log_dir, "loss_plot.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"\n✓ Saved loss plot to {plot_path}", flush=True)
            
        except Exception as e:
            print(f"Error plotting losses: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()


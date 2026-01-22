import torch
from torch import nn, Tensor
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from typing import Any, List, Tuple
from transformers import get_scheduler
import numpy as np
from omegaconf import DictConfig, OmegaConf
import io
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor
from typing import List, Tuple, Union
import random
import os
import numpy as np

from .resnet import resnet18_1D, resnet34_1D
from .loss_function import get_loss_function
import yaml

class CompeleteModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.__initialize_modules(config)
    

    def __initialize_modules(self, config: DictConfig):
        # self.feat_extracter = resnet18_1D(**config.hyperparameters.feat_extracter)
        self.feat_extracter = resnet34_1D(**config.hyperparameters.feat_extracter)
        classifier_cfg = config.hyperparameters.classifier
        in_dim = config.hyperparameters.feat_extracter.feat_dim
        hidden_dims = getattr(classifier_cfg, "hidden_dims", [])
        dropout_p = getattr(classifier_cfg, "dropout_p", 0.0)
        activation_name = getattr(classifier_cfg, "activation", "relu").lower()

        if activation_name == "gelu":
            activation_cls = nn.GELU
        elif activation_name == "leaky_relu":
            activation_cls = lambda: nn.LeakyReLU(negative_slope=getattr(classifier_cfg, "negative_slope", 0.01))
        else:
            activation_cls = nn.ReLU

        layers: List[nn.Module] = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation_cls())
            if dropout_p and dropout_p > 0.0:
                layers.append(nn.Dropout(dropout_p))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, classifier_cfg.num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x:Tensor, sqi=None):
        feat = self.feat_extracter(x, sqi)
        logits = self.classifier(feat)
        
        return logits
    

class EctopicsClassifier(pl.LightningModule):
    def __init__(self, 
                 task: str="binary", 
                 num_classes: int=2, 
                 lr: float=0.0001, 
                 weight_decay: float=0.0001,
                 loss_name: str="cross_entropy",
                 use_lr_scheduler: bool=True,
                 lr_warmup_ratio: float = 0.1,
                 device: str="cuda",
                 total_training_steps: int=1000,
                 config: DictConfig=None,
                 training_config=None,
                 class_weights: List[float] = None,
                 ectopic_threshold: float = None,
                 **kwargs):
        super().__init__()

        # save training config to logs
        if training_config is not None:
            if hasattr(training_config, 'keys'):  # DictConfig or dict
                config_dict = OmegaConf.to_container(training_config, resolve=True)
            else:
                # If it’s still a string path (for backward compatibility)
                with open(training_config, "r") as f:
                    config_dict = yaml.safe_load(f)
            self.save_hyperparameters(config_dict)
        else:
            self.save_hyperparameters()

        self.task = task
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_name = loss_name
        self.use_lr_scheduler = use_lr_scheduler
        self.warmup_ratio = lr_warmup_ratio
        self.class_weights = class_weights
        # Handle OmegaConf null/None values properly
        # OmegaConf converts YAML 'null' to None, but we should handle both cases
        if ectopic_threshold is None:
            self.ectopic_threshold = None
        elif isinstance(ectopic_threshold, str) and ectopic_threshold.lower() in ['null', 'none']:
            self.ectopic_threshold = None
        elif hasattr(ectopic_threshold, '__class__') and 'None' in str(type(ectopic_threshold)):
            # Handle OmegaConf None type if it exists
            self.ectopic_threshold = None
        else:
            try:
                self.ectopic_threshold = float(ectopic_threshold)
            except (ValueError, TypeError):
                print(f"Warning: Could not convert ectopic_threshold '{ectopic_threshold}' to float. Using None (standard argmax).", flush=True)
                self.ectopic_threshold = None
        
        # Debug print
        print(f"Ectopic threshold set to: {self.ectopic_threshold} (using {'standard argmax' if self.ectopic_threshold is None else 'threshold-based prediction'})", flush=True)
    

        if device == "cuda" and torch.cuda.is_available():
            self.device_type = torch.device("cuda")
        else:
            self.device_type = torch.device("cpu")
        
        self.total_steps = total_training_steps
        self.config = config

        self.metrics_lst = []
        for metric in self.config.metrics:
            if self.config.metrics[metric]:
                self.metrics_lst.append(metric)

        print("Loss Function: ", self.loss_name, flush=True)
        print("Metrics: ", self.config.metrics, flush=True)

        self.model = CompeleteModel(self.config)
        
        # Get loss function
        if self.loss_name in ['bce', 'cross_entropy', 'ce']:
            self.loss_fn = get_loss_function(self.loss_name)
        else:
            raise ValueError(f"Invalid loss function: {self.loss_name}. Available: ['bce', 'cross_entropy', 'ce']")
        
        self.metrics = nn.ModuleDict({
            "metrics_train": nn.ModuleDict({}),
            "metrics_valid": nn.ModuleDict({}),
            "metrics_test": nn.ModuleDict({})
        })

        for phase in ["train", "valid", "test"]:
            for metric in self.config.metrics:
                if metric == "accuracy":
                    self.metrics["metrics_" + phase][metric] = torchmetrics.Accuracy(
                        self.task, num_classes=self.num_classes, average="none"
                    )
                elif metric == "cf_matrix":
                    self.metrics["metrics_" + phase][metric] = torchmetrics.ConfusionMatrix(
                        self.task, num_classes=self.num_classes
                    )
                elif metric == "f1":
                    # Add both macro-averaged and per-class F1
                    self.metrics["metrics_" + phase][metric] = torchmetrics.F1Score(
                        self.task, num_classes=self.num_classes, average="macro"
                    )
                    # Per-class F1
                    self.metrics["metrics_" + phase][f"{metric}_per_class"] = torchmetrics.F1Score(
                        self.task, num_classes=self.num_classes, average="none"
                    )
                elif metric == "specificity":
                    # Macro-averaged specificity
                    self.metrics["metrics_" + phase][metric] = torchmetrics.Specificity(
                        self.task, num_classes=self.num_classes, average="macro"
                    )
                    # Per-class specificity
                    self.metrics["metrics_" + phase][f"{metric}_per_class"] = torchmetrics.Specificity(
                        self.task, num_classes=self.num_classes, average="none"
                    )
                elif metric == "AUC":
                    # Macro-averaged AUC
                    self.metrics["metrics_" + phase][metric] = torchmetrics.AUROC(
                        self.task, num_classes=self.num_classes, average="macro"
                    )
                    # Per-class AUC
                    self.metrics["metrics_" + phase][f"{metric}_per_class"] = torchmetrics.AUROC(
                        self.task, num_classes=self.num_classes, average="none"
                    )
                elif metric == "sensitivity":
                    if self.task == "binary":
                        self.metrics["metrics_" + phase][metric] = torchmetrics.Recall(task="binary")
                    else:
                        # Macro-averaged recall
                        self.metrics["metrics_" + phase][metric] = torchmetrics.Recall(
                            task="multiclass", num_classes=self.num_classes, average="macro"
                        )
                        # Per-class recall
                        self.metrics["metrics_" + phase][f"{metric}_per_class"] = torchmetrics.Recall(
                            task="multiclass", num_classes=self.num_classes, average="none"
                        )
                elif metric == "ppv":
                    if self.task == "binary":
                        self.metrics["metrics_" + phase][metric] = torchmetrics.Precision(task="binary")
                    else:
                        # Macro-averaged precision
                        self.metrics["metrics_" + phase][metric] = torchmetrics.Precision(
                            task="multiclass", num_classes=self.num_classes, average="macro"
                        )
                        # Per-class precision
                        self.metrics["metrics_" + phase][f"{metric}_per_class"] = torchmetrics.Precision(
                            task="multiclass", num_classes=self.num_classes, average="none"
                        )

        
        self.step_losses = {"train": [], "valid": [], "test": []}
        
        # Configure misclassified samples collection from training config
        if self.config is not None and hasattr(self.config, 'keys'):
            # Access misclassified_samples config using OmegaConf
            try:
                misclassified_cfg = self.config.misclassified_samples
                self.collect_misclassified_samples = OmegaConf.select(misclassified_cfg, 'enable', default=False)
                self.max_misclassified_to_store = OmegaConf.select(misclassified_cfg, 'max_samples', default=100)
            except (AttributeError, KeyError):
                # Default values if config section doesn't exist
                self.collect_misclassified_samples = False
                self.max_misclassified_to_store = 100
        else:
            # Default values if config not available
            self.collect_misclassified_samples = False
            self.max_misclassified_to_store = 100
        
        # Store misclassified samples for visualization
        self.misclassified_samples = {"test": []}
        
        # Store test predictions, targets, logits, and SQI for stratified metrics
        self.test_predictions = []
        self.test_targets = []
        self.test_logits = []
        self.test_sqi = []
        # Store embeddings before and after attention
        self.test_z_backbone = []
        self.test_z_att = []
        
        # Store per-subject validation data for macro-average metrics
        # Dictionary: {subject_id: {'losses': [], 'predictions': [], 'targets': []}}
        self.valid_per_subject = {}
        
        # Store per-subject training data for macro-average metrics
        # Dictionary: {subject_id: {'losses': [], 'predictions': [], 'targets': []}}
        self.train_per_subject = {}
        
        if self.collect_misclassified_samples:
            print(f"Misclassified samples collection: ENABLED (collecting ALL samples, visualizing up to {self.max_misclassified_to_store} randomly selected samples)", flush=True)
        else:
            print("Misclassified samples collection: DISABLED", flush=True)

    def configure_optimizers(self):
    
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.use_lr_scheduler:
            scheduler = {
                "scheduler": get_scheduler(
                    "polynomial",
                    optimizer,
                    num_warmup_steps=round(self.warmup_ratio * self.total_steps),
                    num_training_steps=self.total_steps,
                ),
                "interval": "step",
                "frequency": 1,
            }
            return [optimizer], [scheduler]

        return optimizer

    def forward(self, x: Tensor, sqi_sample: Tensor = None):
        return self.model(x, sqi=sqi_sample)

    def predict_with_ectopic_threshold(self, logits: Tensor):
        """
        Predict with tunable decision rule for class 'ectopic' (class 1).
        
        Rule:
          if P(ectopic) >= threshold -> predict 1 (ectopic)
          else -> argmax between class 0 and 2
        
        Args:
            logits: [B, 3] tensor of logits
            
        Returns:
            preds: [B] tensor of predictions
        """
        # Use standard argmax if threshold is not set (None, null, or missing)
        if self.ectopic_threshold is None:
            return torch.argmax(logits, dim=1)
        
        probs = F.softmax(logits, dim=1)
        p_ectopic = probs[:, 1]  # Probability of class 1 (ectopic)
        
        preds = torch.empty(logits.size(0), dtype=torch.long, device=logits.device)
        
        # Default: choose between class 0 and 2
        logits_02 = logits[:, [0, 2]]
        preds_02 = torch.argmax(logits_02, dim=1)
        preds[preds_02 == 0] = 0  # If argmax is 0 in [0,2], predict class 0
        preds[preds_02 == 1] = 2  # If argmax is 1 in [0,2], predict class 2
        
        # Override to class 1 (ectopic) when confident enough
        preds[p_ectopic >= self.ectopic_threshold] = 1
        
        return preds

    def plot_confusion_matrix(self, matrix):
        # Row-wise normalization (each row sums to 1) so colors reflect per-class proportions
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.to(dtype=torch.float32)
            row_sums = matrix.sum(dim=1, keepdim=True)
            # Avoid division by zero: only divide rows with positive sum
            matrix = torch.where(row_sums > 0, matrix / row_sums, matrix)
            data = matrix.detach().cpu().numpy()
        else:
            data = np.asarray(matrix, dtype=np.float32)
            row_sums = data.sum(axis=1, keepdims=True)
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                data = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums>0)

        fig, ax = plt.subplots()
        cax = ax.matshow(data, vmin=0.0, vmax=1.0, cmap='Blues')
        fig.colorbar(cax)
    
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        plt.close(fig)  # ensure to close the figure to free memory
        buf.seek(0)
    
        image = Image.open(buf)
        image_tensor = ToTensor()(image)
    
        return image_tensor

    def visualize_misclassified_samples(self, phase: str):
        """Visualize misclassified samples and save to file"""
        if not self.misclassified_samples[phase]:
            return
        
        samples = self.misclassified_samples[phase]
        n_samples = len(samples)
        
        # Randomly select up to max_misclassified_to_store samples for visualization
        n_plots = min(n_samples, self.max_misclassified_to_store)
        if n_samples > n_plots:
            # Randomly sample indices
            random.seed(42)  # For reproducibility
            selected_indices = random.sample(range(n_samples), n_plots)
            samples_to_plot = [samples[i] for i in selected_indices]
        else:
            samples_to_plot = samples
        
        # Determine whether ECG traces are available
        has_ecg = any('ecg' in sample for sample in samples_to_plot)

        if has_ecg:
            fig, axes = plt.subplots(n_plots, 2, figsize=(18, 3.5 * n_plots))
            if n_plots == 1:
                axes = axes.reshape(1, 2)
        else:
            n_cols = 4
            n_rows = (n_plots + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 4 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()
        
        label_names = ['Normal', 'Ectopic', 'VT']
        
        def _to_1d(signal_array):
            """Convert signal array to 1D for visualization.
            For multi-channel input (e.g., 5 channels: PPG + 4 FP masks), 
            extract the first channel (PPG) for visualization.
            """
            if isinstance(signal_array, torch.Tensor):
                signal_array = signal_array.detach().cpu().numpy()
            if signal_array.ndim == 1:
                return signal_array
            if signal_array.ndim == 2:
                # Handle multi-channel input: (channels, signal_length) or (signal_length, channels)
                if signal_array.shape[0] == 1:
                    # Single channel: (1, signal_length) -> (signal_length,)
                    return signal_array[0]
                if signal_array.shape[1] == 1:
                    # Single channel: (signal_length, 1) -> (signal_length,)
                    return signal_array[:, 0]
                # Multi-channel: extract first channel (PPG channel)
                # Shape is (channels, signal_length), return first channel
                return signal_array[0]
            return signal_array.reshape(-1)

        for i in range(n_plots):
            sample = samples_to_plot[i]
            signal = _to_1d(sample['input'])

            if has_ecg:
                ppg_ax = axes[i, 0]
                ppg_ax.plot(signal, linewidth=1.5)
                prob_str = ', '.join([f'{p:.3f}' for p in sample["probabilities"]])
                ppg_ax.set_title(
                    f'True: {label_names[sample["true_label"]]}, '
                    f'Pred: {label_names[sample["prediction"]]}'
                    f'\nProb: [{prob_str}]',
                    fontsize=9
                )
                ppg_ax.grid(True, alpha=0.3)
                ppg_ax.set_xlabel('Time')
                ppg_ax.set_ylabel('PPG Amplitude')

                ecg_data = sample.get('ecg')
                if ecg_data is not None:
                    ecg_signal = _to_1d(ecg_data)
                    ecg_ax = axes[i, 1]
                    ecg_ax.plot(ecg_signal, color='tab:red', linewidth=1.2)
                    ecg_ax.set_title('Corresponding ECG', fontsize=9)
                    ecg_ax.grid(True, alpha=0.3)
                    ecg_ax.set_xlabel('Time')
                    ecg_ax.set_ylabel('ECG Amplitude')
                else:
                    axes[i, 1].axis('off')
            else:
                ax = axes[i]
                ax.plot(signal, linewidth=1.5)
                prob_str = ', '.join([f'{p:.3f}' for p in sample["probabilities"]])
                ax.set_title(
                    f'True: {label_names[sample["true_label"]]}, '
                    f'Pred: {label_names[sample["prediction"]]}'
                    f'\nProb: [{prob_str}]',
                    fontsize=9
                )
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Time')
                ax.set_ylabel('Amplitude')

        if not has_ecg:
            # Hide unused subplots
            for i in range(n_plots, len(axes)):
                axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save to file and TensorBoard
        import os
        if hasattr(self, 'logger') and self.logger is not None:
            version = self.logger.version if self.logger.version is not None else ""
            if version:
                save_dir = os.path.join(self.logger.save_dir, self.logger.name, f"version_{version}")
            else:
                save_dir = os.path.join(self.logger.save_dir, self.logger.name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{phase}_misclassified_samples.png")
            
            # Save to file
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nSaved {n_plots} randomly selected misclassified {phase} samples (out of {n_samples} total) to {save_path}", flush=True)
            
            # Also log to TensorBoard if available
            if hasattr(self.logger, 'experiment') and self.logger.experiment is not None:
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                
                image = Image.open(buf)
                image_tensor = ToTensor()(image)
                self.logger.experiment.add_image(f"{phase}_misclassified_samples", image_tensor, global_step=self.current_epoch)
        else:
            # Fallback: save to current directory
            save_path = f"{phase}_misclassified_samples.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nSaved {n_plots} randomly selected misclassified {phase} samples (out of {n_samples} total) to {save_path}", flush=True)
        
        plt.close(fig)

    def save_misclassified_filenames(self, phase: str):
        """Save filenames of misclassified samples to a text file"""
        if not self.misclassified_samples[phase]:
            return
        
        samples = self.misclassified_samples[phase]
        filenames = []
        
        for sample in samples:
            if 'filename' in sample:
                filenames.append(str(sample['filename']))
        
        if not filenames:
            print(f"No filenames found in misclassified {phase} samples.", flush=True)
            return
        
        # Determine save directory
        import os
        if hasattr(self, 'logger') and self.logger is not None:
            version = self.logger.version if self.logger.version is not None else ""
            if version:
                save_dir = os.path.join(self.logger.save_dir, self.logger.name, f"version_{version}")
            else:
                save_dir = os.path.join(self.logger.save_dir, self.logger.name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{phase}_misclassified_filenames.txt")
        else:
            # Fallback: save to current directory
            save_path = f"{phase}_misclassified_filenames.txt"
        
        # Save filenames to file
        with open(save_path, 'w') as f:
            f.write(f"Misclassified {phase} samples - Filenames\n")
            f.write(f"Total misclassified samples: {len(samples)}\n")
            f.write(f"Samples with filenames: {len(filenames)}\n")
            f.write("=" * 80 + "\n\n")
            for i, filename in enumerate(filenames, 1):
                # Get additional info if available
                sample = samples[i-1]
                true_label = sample.get('true_label', 'N/A')
                prediction = sample.get('prediction', 'N/A')
                label_names = ['Normal', 'Ectopic', 'VT']
                true_label_name = label_names[true_label] if isinstance(true_label, int) and 0 <= true_label < len(label_names) else str(true_label)
                pred_label_name = label_names[prediction] if isinstance(prediction, int) and 0 <= prediction < len(label_names) else str(prediction)
                f.write(f"{i}. {filename} (True: {true_label_name}, Pred: {pred_label_name})\n")
        
        print(f"\nSaved {len(filenames)} misclassified {phase} filenames to {save_path}", flush=True)

    def save_test_confusion_matrix(self, matrix):
        """Save and report test confusion matrix to file and console"""
        # Convert to numpy if tensor
        if isinstance(matrix, torch.Tensor):
            cm = matrix.detach().cpu().numpy()
        else:
            cm = np.asarray(matrix)
        
        # Class labels
        label_names = ['Normal', 'Ectopic', 'VT']
        
        # Print confusion matrix to console
        print("\n" + "=" * 80, flush=True)
        print("TEST CONFUSION MATRIX", flush=True)
        print("=" * 80, flush=True)
        print(f"\n{'':<12}", end="", flush=True)
        for i, name in enumerate(label_names):
            print(f"{name:>12}", end="", flush=True)
        print("", flush=True)
        
        for i, name in enumerate(label_names):
            print(f"{name:<12}", end="", flush=True)
            for j in range(len(label_names)):
                print(f"{int(cm[i, j]):>12}", end="", flush=True)
            print("", flush=True)
        
        # Print row-wise percentages
        print("\nRow-wise percentages (True class distribution):", flush=True)
        print(f"{'':<12}", end="", flush=True)
        for i, name in enumerate(label_names):
            print(f"{name:>12}", end="", flush=True)
        print("", flush=True)
        
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_percent = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums>0) * 100
        
        for i, name in enumerate(label_names):
            print(f"{name:<12}", end="", flush=True)
            for j in range(len(label_names)):
                print(f"{cm_percent[i, j]:>11.1f}%", end="", flush=True)
            print("", flush=True)
        
        print("=" * 80 + "\n", flush=True)
        
        # Determine save directory
        if hasattr(self, 'logger') and self.logger is not None:
            version = self.logger.version if self.logger.version is not None else ""
            if version:
                save_dir = os.path.join(self.logger.save_dir, self.logger.name, f"version_{version}")
            else:
                save_dir = os.path.join(self.logger.save_dir, self.logger.name)
            os.makedirs(save_dir, exist_ok=True)
        else:
            # Fallback: save to current directory
            save_dir = "."
        
        # Save confusion matrix as numpy array
        npy_path = os.path.join(save_dir, "test_confusion_matrix.npy")
        np.save(npy_path, cm)
        print(f"Saved confusion matrix (numpy) to {npy_path}", flush=True)
        
        # Save confusion matrix as text file
        txt_path = os.path.join(save_dir, "test_confusion_matrix.txt")
        with open(txt_path, 'w') as f:
            f.write("TEST CONFUSION MATRIX\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"{'':<12}")
            for i, name in enumerate(label_names):
                f.write(f"{name:>12}")
            f.write("\n")
            
            for i, name in enumerate(label_names):
                f.write(f"{name:<12}")
                for j in range(len(label_names)):
                    f.write(f"{int(cm[i, j]):>12}")
                f.write("\n")
            
            f.write("\nRow-wise percentages (True class distribution):\n")
            f.write(f"{'':<12}")
            for i, name in enumerate(label_names):
                f.write(f"{name:>12}")
            f.write("\n")
            
            for i, name in enumerate(label_names):
                f.write(f"{name:<12}")
                for j in range(len(label_names)):
                    f.write(f"{cm_percent[i, j]:>11.1f}%")
                f.write("\n")
        
        print(f"Saved confusion matrix (text) to {txt_path}", flush=True)
        
        # Save confusion matrix as image
        img_path = os.path.join(save_dir, "test_confusion_matrix.png")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Use absolute counts for display
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(int(cm[i, j]), 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=label_names,
               yticklabels=label_names,
               ylabel='True Label',
               xlabel='Predicted Label',
               title='Test Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved confusion matrix (image) to {img_path}", flush=True)

    def update_metrics(self, outputs, targets, phase: str = "train", logits: torch.Tensor = None):
        """
        Update metrics with predictions and optionally logits.
        
        Args:
            outputs: Class predictions (argmax of logits) - shape [B]
            targets: Ground truth labels - shape [B]
            phase: Phase name ("train", "valid", "test")
            logits: Optional logits/probabilities - shape [B, num_classes] (needed for AUC)
        """
        for k in self.config.metrics:
            if k in self.metrics["metrics_" + phase]:
                # AUC needs logits/probabilities, not class predictions
                if k == "AUC" and logits is not None:
                    self.metrics["metrics_" + phase][k].update(logits, targets)
                else:
                    # Other metrics use class predictions
                    self.metrics["metrics_" + phase][k].update(outputs, targets)
            # Also update per-class metrics if they exist
            if f"{k}_per_class" in self.metrics["metrics_" + phase]:
                # Per-class AUC also needs logits
                if k == "AUC" and logits is not None:
                    self.metrics["metrics_" + phase][f"{k}_per_class"].update(logits, targets)
                else:
                    # Other per-class metrics use class predictions
                    self.metrics["metrics_" + phase][f"{k}_per_class"].update(outputs, targets)

    def reset_metrics(self, phase: str = "train"):
        for k in self.config.metrics:
            if k in self.metrics["metrics_" + phase]:
                self.metrics["metrics_" + phase][k].reset()
            # Also reset per-class metrics if they exist
            if f"{k}_per_class" in self.metrics["metrics_" + phase]:
                self.metrics["metrics_" + phase][f"{k}_per_class"].reset()
    
    def _compute_sqi_stratified_metrics(self):
        """Compute metrics stratified by SQI bins: Low (<0.3), Medium (0.3-0.7), High (>0.7)"""
        # Collect all batches with valid SQI
        test_preds = []
        test_targets = []
        test_logits = []
        test_sqi = []
        
        for i, sqi_batch in enumerate(self.test_sqi):
            if sqi_batch is not None:
                test_preds.append(self.test_predictions[i])
                test_targets.append(self.test_targets[i])
                test_logits.append(self.test_logits[i])
                test_sqi.append(sqi_batch)
        
        if not test_sqi:
            print("Warning: No valid SQI values found for stratified metrics.", flush=True)
            return
        
        # Concatenate all test data
        all_preds = torch.cat(test_preds, dim=0)
        all_targets = torch.cat(test_targets, dim=0)
        all_logits = torch.cat(test_logits, dim=0)
        all_sqi = torch.cat(test_sqi, dim=0)
        
        # Ensure all tensors have the same length
        min_len = min(len(all_preds), len(all_targets), len(all_logits), len(all_sqi))
        all_preds = all_preds[:min_len]
        all_targets = all_targets[:min_len]
        all_logits = all_logits[:min_len]
        all_sqi = all_sqi[:min_len]
        
        # Save test data to file for later analysis
        save_dir = os.path.join(self.logger.save_dir, self.logger.name, f"version_{self.logger.version}")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "test_sqi_data.npz")
        
        # Convert tensors to numpy and save
        np.savez(
            save_path,
            predictions=all_preds.detach().cpu().numpy(),
            targets=all_targets.detach().cpu().numpy(),
            logits=all_logits.detach().cpu().numpy(),
            sqi=all_sqi.detach().cpu().numpy()
        )
        print(f"Saved test data for SQI analysis to {save_path}", flush=True)
        
        # Define SQI bins
        low_mask = all_sqi < 30
        medium_mask = (all_sqi >= 30) & (all_sqi <= 70)
        high_mask = all_sqi > 70
        
        sqi_bins = {
            "low": (low_mask, "Low SQI (s < 30)"),
            "medium": (medium_mask, "Medium SQI (30 ≤ s ≤ 70)"),
            "high": (high_mask, "High SQI (s > 70)")
        }
        
        # Compute metrics for each bin
        for bin_name, (mask, bin_label) in sqi_bins.items():
            n_samples = mask.sum().item()
            if n_samples == 0:
                print(f"Warning: No samples in {bin_label}. Skipping metrics.", flush=True)
                continue
            
            bin_preds = all_preds[mask]
            bin_targets = all_targets[mask]
            
            # Create temporary metric objects for this bin
            bin_metrics = {}
            
            for metric_name in self.config.metrics:
                if not self.config.metrics[metric_name]:
                    continue
                    
                try:
                    if metric_name == "accuracy":
                        metric_obj = torchmetrics.Accuracy(
                            self.task, num_classes=self.num_classes, average="none"
                        ).to(bin_preds.device)
                        metric_obj.update(bin_preds, bin_targets)
                        acc = metric_obj.compute()
                        bin_metrics["accuracy"] = acc.mean().item() if acc.numel() > 1 else acc.item()
                    
                    elif metric_name == "f1":
                        metric_obj = torchmetrics.F1Score(
                            self.task, num_classes=self.num_classes
                        ).to(bin_preds.device)
                        metric_obj.update(bin_preds, bin_targets)
                        bin_metrics["f1"] = metric_obj.compute().item()
                    
                    elif metric_name == "specificity":
                        metric_obj = torchmetrics.Specificity(
                            self.task, num_classes=self.num_classes
                        ).to(bin_preds.device)
                        metric_obj.update(bin_preds, bin_targets)
                        bin_metrics["specificity"] = metric_obj.compute().item()
                    
                    elif metric_name == "AUC":
                        # For AUC, we need logits (probabilities)
                        bin_logits_subset = all_logits[mask]
                        if self.task == "binary":
                            bin_logits_subset = bin_logits_subset[:, 1]  # Take logit for positive class
                        metric_obj = torchmetrics.AUROC(
                            self.task, num_classes=self.num_classes if self.task != "binary" else None
                        ).to(bin_logits_subset.device)
                        metric_obj.update(bin_logits_subset, bin_targets)
                        bin_metrics["AUC"] = metric_obj.compute().item()
                    
                    elif metric_name == "sensitivity":
                        if self.task == "binary":
                            metric_obj = torchmetrics.Recall(task="binary").to(bin_preds.device)
                        else:
                            metric_obj = torchmetrics.Recall(
                                task="multiclass", num_classes=self.num_classes, average="macro"
                            ).to(bin_preds.device)
                        metric_obj.update(bin_preds, bin_targets)
                        bin_metrics["sensitivity"] = metric_obj.compute().item()
                    
                    elif metric_name == "ppv":
                        if self.task == "binary":
                            metric_obj = torchmetrics.Precision(task="binary").to(bin_preds.device)
                        else:
                            metric_obj = torchmetrics.Precision(
                                task="multiclass", num_classes=self.num_classes, average="macro"
                            ).to(bin_preds.device)
                        metric_obj.update(bin_preds, bin_targets)
                        bin_metrics["ppv"] = metric_obj.compute().item()
                    
                    elif metric_name == "cf_matrix":
                        metric_obj = torchmetrics.ConfusionMatrix(
                            self.task, num_classes=self.num_classes
                        ).to(bin_preds.device)
                        metric_obj.update(bin_preds, bin_targets)
                        cm = metric_obj.compute()
                        bin_metrics["confusion_matrix"] = cm
                
                except Exception as e:
                    print(f"Warning: Could not compute {metric_name} for {bin_label}: {e}", flush=True)
            
            # Log metrics for this bin
            log_items = []
            for metric_key, metric_value in bin_metrics.items():
                if metric_key == "confusion_matrix":
                    log_items.append((f"sqi_{bin_name}_confusion_matrix", metric_value))
                else:
                    log_items.append((f"sqi_{bin_name}_{metric_key}", metric_value))
            
            # Also log sample count
            log_items.append((f"sqi_{bin_name}_n_samples", float(n_samples)))
            
            self.log_all(
                items=log_items,
                phase="test",
                prog_bar=False,
                sync_dist_group=False,
            )
            
            # Print summary
            print(f"\n{bin_label} (n={n_samples}):", flush=True)
            for metric_key, metric_value in bin_metrics.items():
                if metric_key != "confusion_matrix":
                    print(f"  {metric_key}: {metric_value:.4f}", flush=True)
    
    def save_embeddings_with_labels_sqi(self):
        """Save embeddings (before and after attention) with corresponding true labels and SQI values"""
        if not self.test_z_backbone or all(z is None for z in self.test_z_backbone):
            print("Warning: No embeddings found to save.", flush=True)
            return
        
        # Get save directory
        if self.logger is not None:
            version = self.logger.version if self.logger.version is not None else ""
            if version:
                save_dir = os.path.join(self.logger.save_dir, self.logger.name, f"version_{version}")
            else:
                save_dir = os.path.join(self.logger.save_dir, self.logger.name)
        else:
            save_dir = "."
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Collect all embeddings, labels, and SQI
        z_backbone_list = []
        z_att_list = []
        labels_list = []
        sqi_list = []
        
        for i in range(len(self.test_z_backbone)):
            if self.test_z_backbone[i] is not None:
                z_backbone_list.append(self.test_z_backbone[i])
                labels_list.append(self.test_targets[i])
                
                # Handle z_att (may be None if attention is not used)
                if self.test_z_att[i] is not None:
                    z_att_list.append(self.test_z_att[i])
                else:
                    # Create dummy tensor with same shape as z_backbone
                    z_att_list.append(torch.zeros_like(self.test_z_backbone[i]))
                
                # Handle SQI
                if self.test_sqi[i] is not None:
                    sqi_list.append(self.test_sqi[i])
                else:
                    # Create dummy SQI values (e.g., -1 to indicate missing)
                    batch_size = self.test_z_backbone[i].shape[0]
                    sqi_list.append(torch.full((batch_size,), -1.0))
        
        if not z_backbone_list:
            print("Warning: No valid embeddings found to save.", flush=True)
            return
        
        # Concatenate all batches
        all_z_backbone = torch.cat(z_backbone_list, dim=0).numpy()  # (N, C)
        all_z_att = torch.cat(z_att_list, dim=0).numpy()  # (N, C)
        all_labels = torch.cat(labels_list, dim=0).numpy()  # (N,)
        all_sqi = torch.cat(sqi_list, dim=0).numpy()  # (N,)
        
        # Ensure all arrays have the same length
        min_len = min(len(all_z_backbone), len(all_z_att), len(all_labels), len(all_sqi))
        all_z_backbone = all_z_backbone[:min_len]
        all_z_att = all_z_att[:min_len]
        all_labels = all_labels[:min_len]
        all_sqi = all_sqi[:min_len]
        
        # Save as NPZ file
        npz_path = os.path.join(save_dir, "test_embeddings.npz")
        np.savez(
            npz_path,
            z_backbone=all_z_backbone,  # Embeddings before attention (N, C)
            z_att=all_z_att,  # Embeddings after attention (N, C)
            labels=all_labels,  # True labels (N,)
            sqi=all_sqi  # SQI values (N,)
        )
        print(f"Saved embeddings with labels and SQI to {npz_path}", flush=True)
        print(f"  - z_backbone shape: {all_z_backbone.shape}", flush=True)
        print(f"  - z_att shape: {all_z_att.shape}", flush=True)
        print(f"  - labels shape: {all_labels.shape}", flush=True)
        print(f"  - sqi shape: {all_sqi.shape}", flush=True)
        print(f"  - Total samples: {min_len}", flush=True)
    
    def log_all(self, items: List[Tuple[str, Union[float, torch.Tensor]]], phase: str = "train", prog_bar: bool = True, sync_dist_group: bool = False):
        for key, value in items:
            if value is not None:
                # Check if value is a float
                if isinstance(value, float):
                    self.log(f"{phase}_{key}", value, prog_bar=prog_bar, sync_dist_group=sync_dist_group)
                # Check if value is a tensor
                elif isinstance(value, torch.Tensor):
                    if len(value.shape) == 0:  # Scalar tensor
                        self.log(f"{phase}_{key}", value, prog_bar=prog_bar, sync_dist_group=sync_dist_group)
                    elif len(value.shape) == 2:  # 2D tensor, assume confusion matrix and log as image
                        image_tensor = self.plot_confusion_matrix(value)
                        self.logger.experiment.add_image(f"{phase}_{key}", image_tensor, global_step=self.current_epoch)

    def training_step(self, batch, batch_idx):
        
        # Support multiple formats:
        # (x, targets) - no SQI, no id0
        # (x, targets, sqi) - scalar SQI only
        # (x, targets, sqi_sample) - temporal SQI only  
        # (x, targets, sqi, sqi_sample) - both scalar and temporal SQI
        # (x, targets, id0) - with id0
        # (x, targets, sqi, id0) - with sqi and id0
        # (x, targets, sqi_sample, id0) - with sqi_sample and id0
        # (x, targets, sqi, sqi_sample, id0) - with all
        sqi = None
        sqi_sample = None
        id0_batch = None
        
        if isinstance(batch, (list, tuple)):
            # id0 is always last if present (based on BalancedTrainDataset.__getitem__)
            # Check if last item is id0 (string or object array)
            last_item = batch[-1] if len(batch) > 2 else None
            is_id0 = (last_item is not None and 
                     (isinstance(last_item, str) or 
                      (isinstance(last_item, np.ndarray) and last_item.dtype == object) or
                      (isinstance(last_item, (list, tuple)) and len(last_item) > 0 and isinstance(last_item[0], str))))
            
            if is_id0:
                # Last item is id0, extract it and process the rest
                id0_batch = batch[-1]
                batch_without_id0 = batch[:-1]
            else:
                batch_without_id0 = batch
            
            # Process batch without id0 (same as before)
            if len(batch_without_id0) == 2:
                x, targets = batch_without_id0
            elif len(batch_without_id0) == 3:
                x, targets, third_item = batch_without_id0
                # Determine if third_item is scalar sqi or temporal sqi_sample
                if not isinstance(third_item, torch.Tensor):
                    third_item = torch.from_numpy(third_item).float()
                third_item = third_item.to(self.device_type)
                if third_item.ndim == 1:
                    sqi = third_item  # Scalar SQI
                elif third_item.ndim == 2:
                    sqi_sample = third_item  # Temporal SQI
            elif len(batch_without_id0) == 4:
                # (x, targets, sqi, sqi_sample)
                x, targets, sqi_raw, sqi_sample_raw = batch_without_id0
                # Convert to tensors if needed
                if not isinstance(sqi_raw, torch.Tensor):
                    sqi_raw = torch.from_numpy(sqi_raw).float()
                if not isinstance(sqi_sample_raw, torch.Tensor):
                    sqi_sample_raw = torch.from_numpy(sqi_sample_raw).float()
                sqi = sqi_raw.to(self.device_type)
                sqi_sample = sqi_sample_raw.to(self.device_type)
            else:
                x, targets = batch_without_id0[0], batch_without_id0[1]
        else:
            x, targets = batch
        
        output_logits = self(x, sqi_sample=sqi_sample)
        # preds = self.predict_with_ectopic_threshold(output_logits)
        preds = torch.argmax(output_logits, dim=1)


        if self.loss_name in ["bce", "cross_entropy", "ce"]:
            # Get per-sample task loss
            # Pass class_weights if available for 3-class imbalanced classification
            per_sample_loss = self.loss_fn(
                targets, output_logits, self.device_type, reduction="none", 
                class_weights=self.class_weights
            )
            
            # # Apply SQI weighting if available
            # if sqi is not None:
            #     w = torch.clamp(sqi, min=0.2, max=1.0)
            #     per_sample_loss = w * per_sample_loss
            
            # Compute macro-averaged loss (average of per-subject average losses) if id0 is available
            if id0_batch is not None:
                # Convert id0_batch to list if needed
                if isinstance(id0_batch, np.ndarray):
                    id0_list = id0_batch.tolist()
                elif isinstance(id0_batch, torch.Tensor):
                    id0_list = id0_batch.cpu().numpy().tolist()
                elif isinstance(id0_batch, (list, tuple)):
                    id0_list = list(id0_batch)
                else:
                    id0_list = [id0_batch]
                
                # Group per-sample losses by subject and compute per-subject averages
                # Use a dictionary to track which samples belong to which subject in this batch
                subject_losses = {}  # {subject_id_str: list of loss tensors (with gradients)}
                subject_indices = {}  # {subject_id_str: list of sample indices}
                
                for i, subject_id in enumerate(id0_list):
                    subject_id_str = str(subject_id)
                    if subject_id_str not in subject_indices:
                        subject_indices[subject_id_str] = []
                        subject_losses[subject_id_str] = []
                    subject_indices[subject_id_str].append(i)
                    subject_losses[subject_id_str].append(per_sample_loss[i])
                
                # Compute per-subject average losses (each subject contributes equally to macro loss)
                per_subject_avg_losses = []
                for subject_id_str, loss_list in subject_losses.items():
                    if len(loss_list) > 0:
                        # Average losses for this subject (with gradients preserved)
                        subject_avg_loss = torch.stack(loss_list).mean()
                        per_subject_avg_losses.append(subject_avg_loss)
                
                # Macro loss: average of per-subject average losses
                if len(per_subject_avg_losses) > 0:
                    loss = torch.stack(per_subject_avg_losses).mean()
                else:
                    # Fallback to regular mean if no subjects found
                    loss = per_sample_loss.mean()
                
                # Track per-subject data for logging (move to CPU after loss computation)
                preds_cpu = preds.cpu().numpy()
                targets_cpu = targets.cpu().numpy()
                loss_cpu = per_sample_loss.detach().cpu().numpy()
                
                # Group by subject for tracking
                for i, subject_id in enumerate(id0_list):
                    subject_id_str = str(subject_id)
                    if subject_id_str not in self.train_per_subject:
                        self.train_per_subject[subject_id_str] = {
                            'losses': [],
                            'predictions': [],
                            'targets': []
                        }
                    self.train_per_subject[subject_id_str]['losses'].append(float(loss_cpu[i]))
                    self.train_per_subject[subject_id_str]['predictions'].append(int(preds_cpu[i]))
                    self.train_per_subject[subject_id_str]['targets'].append(int(targets_cpu[i]))
            else:
                # # No id0 available, use regular mean loss
                # if sqi is not None:
                #     # Already applied SQI weighting above, now normalize
                #     loss = per_sample_loss.sum() / w.sum()
                # else:
                loss = per_sample_loss.mean()
        else:
            raise ValueError(f"Invalid loss function: {self.loss_name}")
        
        # Pass logits for AUC metric, predictions for other metrics
        self.update_metrics(preds, targets, "train", logits=output_logits)
          
        self.step_losses["train"].append(loss.item())
        
        return {"loss": loss}

    def on_train_epoch_start(self):
        """Clear per-subject data at the start of training epoch"""
        self.train_per_subject.clear()
    
    def on_train_epoch_end(self):
        """End of the training epoch"""
        avg_loss = sum(self.step_losses["train"]) / len(self.step_losses["train"])

        acc = matrix = f1 = spec = auc = sensitivity = ppv = None
        acc_per_class = f1_per_class = sensitivity_per_class = ppv_per_class = None
        spec_per_class = auc_per_class = None

        if "accuracy" in self.metrics_lst:
            acc = self.metrics["metrics_" + "train"]["accuracy"].compute()
            # For multiclass, accuracy with average="none" returns per-class accuracy
            if self.task == "multiclass" and acc.numel() > 1:
                acc_per_class = acc

        if "cf_matrix" in self.metrics_lst:
            matrix = self.metrics["metrics_" + "train"]["cf_matrix"].compute()

        if "f1" in self.metrics_lst:
            f1 = self.metrics["metrics_" + "train"]["f1"].compute()
            if "f1_per_class" in self.metrics["metrics_" + "train"]:
                f1_per_class = self.metrics["metrics_" + "train"]["f1_per_class"].compute()
        
        if "specificity" in self.metrics_lst:
            spec = self.metrics["metrics_" + "train"]["specificity"].compute()
            if "specificity_per_class" in self.metrics["metrics_" + "train"]:
                spec_per_class = self.metrics["metrics_" + "train"]["specificity_per_class"].compute()

        if "AUC"  in self.metrics_lst:
            auc = self.metrics["metrics_" + "train"]["AUC"].compute()
            if "AUC_per_class" in self.metrics["metrics_" + "train"]:
                auc_per_class = self.metrics["metrics_" + "train"]["AUC_per_class"].compute()

        if "sensitivity" in self.metrics_lst:
            sensitivity = self.metrics["metrics_" + "train"]["sensitivity"].compute()
            if "sensitivity_per_class" in self.metrics["metrics_" + "train"]:
                sensitivity_per_class = self.metrics["metrics_" + "train"]["sensitivity_per_class"].compute()

        if "ppv" in self.metrics_lst:
            ppv = self.metrics["metrics_" + "train"]["ppv"].compute()
            if "ppv_per_class" in self.metrics["metrics_" + "train"]:
                ppv_per_class = self.metrics["metrics_" + "train"]["ppv_per_class"].compute()
        
        # Build log items
        log_items = [
            ("loss", avg_loss),
            ("accuracy", acc.mean() if acc_per_class is not None else acc),
            ("specificity", spec),
            ("AUC", auc),
            ("sensitivity", sensitivity),
            ("ppv", ppv),
            ("confusion_matrix", matrix),
            ("f1", f1),
        ]
        
        # Add per-class metrics
        if acc_per_class is not None:
            for i, acc_val in enumerate(acc_per_class):
                log_items.append((f"accuracy_class_{i}", acc_val))
        
        if f1_per_class is not None:
            for i, f1_val in enumerate(f1_per_class):
                log_items.append((f"f1_class_{i}", f1_val))
        
        if sensitivity_per_class is not None:
            for i, sens_val in enumerate(sensitivity_per_class):
                log_items.append((f"sensitivity_class_{i}", sens_val))
        
        if ppv_per_class is not None:
            for i, ppv_val in enumerate(ppv_per_class):
                log_items.append((f"ppv_class_{i}", ppv_val))
        
        if spec_per_class is not None:
            for i, spec_val in enumerate(spec_per_class):
                log_items.append((f"specificity_class_{i}", spec_val))
        
        if auc_per_class is not None:
            for i, auc_val in enumerate(auc_per_class):
                log_items.append((f"AUC_class_{i}", auc_val))
        
        self.log_all(
                items=log_items,
                phase="train",
                prog_bar=True,
                sync_dist_group=False,
            )
        
        # Calculate per-subject macro-average metrics if per-subject data is available
        if len(self.train_per_subject) > 0:
            # Calculate per-subject macro-average loss (each subject counts equally)
            subject_losses = []
            for subject_id, data in self.train_per_subject.items():
                # Average loss for this subject
                subject_avg_loss = np.mean(data['losses'])
                subject_losses.append(subject_avg_loss)
            
            # Macro-average loss across subjects
            macro_avg_loss = np.mean(subject_losses) if len(subject_losses) > 0 else avg_loss
            
            # Calculate per-subject macro-average metrics
            # For each subject, calculate metrics, then average across subjects
            subject_accuracies = []
            subject_f1_scores = []
            subject_sensitivities = []
            subject_specificities = []
            subject_ppvs = []
            
            for subject_id, data in self.train_per_subject.items():
                preds_subject = torch.tensor(data['predictions'], device=self.device_type)
                targets_subject = torch.tensor(data['targets'], device=self.device_type)
                
                # Calculate metrics for this subject
                if "accuracy" in self.metrics_lst:
                    acc_metric = torchmetrics.Accuracy(
                        self.task, num_classes=self.num_classes, average="none"
                    ).to(self.device_type)
                    acc_subject = acc_metric(preds_subject, targets_subject)
                    if acc_subject.numel() > 1:
                        subject_accuracies.append(acc_subject.mean().item())
                    else:
                        subject_accuracies.append(acc_subject.item())
                
                if "f1" in self.metrics_lst:
                    f1_metric = torchmetrics.F1Score(
                        self.task, num_classes=self.num_classes, average="macro"
                    ).to(self.device_type)
                    f1_subject = f1_metric(preds_subject, targets_subject)
                    subject_f1_scores.append(f1_subject.item())
                
                if "sensitivity" in self.metrics_lst:
                    if self.task == "binary":
                        sens_metric = torchmetrics.Recall(task="binary").to(self.device_type)
                    else:
                        sens_metric = torchmetrics.Recall(
                            task="multiclass", num_classes=self.num_classes, average="macro"
                        ).to(self.device_type)
                    sens_subject = sens_metric(preds_subject, targets_subject)
                    subject_sensitivities.append(sens_subject.item())
                
                if "specificity" in self.metrics_lst:
                    spec_metric = torchmetrics.Specificity(
                        self.task, num_classes=self.num_classes, average="macro"
                    ).to(self.device_type)
                    spec_subject = spec_metric(preds_subject, targets_subject)
                    subject_specificities.append(spec_subject.item())
                
                if "ppv" in self.metrics_lst:
                    if self.task == "binary":
                        ppv_metric = torchmetrics.Precision(task="binary").to(self.device_type)
                    else:
                        ppv_metric = torchmetrics.Precision(
                            task="multiclass", num_classes=self.num_classes, average="macro"
                        ).to(self.device_type)
                    ppv_subject = ppv_metric(preds_subject, targets_subject)
                    subject_ppvs.append(ppv_subject.item())
            
            # Calculate macro-averages across subjects
            per_subject_log_items = [
                ("loss_macro", macro_avg_loss),
            ]
            
            if len(subject_accuracies) > 0:
                per_subject_log_items.append(("accuracy_macro", np.mean(subject_accuracies)))
            if len(subject_f1_scores) > 0:
                per_subject_log_items.append(("f1_macro", np.mean(subject_f1_scores)))
            if len(subject_sensitivities) > 0:
                per_subject_log_items.append(("sensitivity_macro", np.mean(subject_sensitivities)))
            if len(subject_specificities) > 0:
                per_subject_log_items.append(("specificity_macro", np.mean(subject_specificities)))
            if len(subject_ppvs) > 0:
                per_subject_log_items.append(("ppv_macro", np.mean(subject_ppvs)))
            
            # Log per-subject macro-average metrics
            self.log_all(
                items=per_subject_log_items,
                phase="train",
                prog_bar=True,
                sync_dist_group=False,
            )
            
            # Clear per-subject data for next epoch
            self.train_per_subject.clear()
        
        self.reset_metrics("train")
        self.step_losses["train"].clear()

    def validation_step(self, batch):
        
        # Support multiple formats:
        # (x, targets) - no SQI, no id0
        # (x, targets, sqi) - scalar SQI (ignored)
        # (x, targets, sqi_sample) - temporal SQI
        # (x, targets, sqi, sqi_sample) - both (use sqi_sample)
        # (x, targets, id0) - with id0
        # (x, targets, sqi, id0) - with sqi and id0
        # (x, targets, sqi_sample, id0) - with sqi_sample and id0
        # (x, targets, sqi, sqi_sample, id0) - with all
        sqi_sample = None
        id0_batch = None
        
        if isinstance(batch, (list, tuple)):
            # id0 is always last if present (based on ValidDataset.__getitem__)
            # Check if last item is id0 (string or object array)
            last_item = batch[-1] if len(batch) > 2 else None
            is_id0 = (last_item is not None and 
                     (isinstance(last_item, str) or 
                      (isinstance(last_item, np.ndarray) and last_item.dtype == object) or
                      (isinstance(last_item, (list, tuple)) and len(last_item) > 0 and isinstance(last_item[0], str))))
            
            if is_id0:
                # Last item is id0, extract it and process the rest
                id0_batch = batch[-1]
                batch_without_id0 = batch[:-1]
            else:
                batch_without_id0 = batch
            
            # Process batch without id0 (same as before)
            if len(batch_without_id0) == 2:
                x, targets = batch_without_id0
            elif len(batch_without_id0) == 3:
                x, targets, third_item = batch_without_id0
                # Convert to tensor if needed
                if not isinstance(third_item, torch.Tensor):
                    third_item = torch.from_numpy(third_item).float()
                third_item = third_item.to(self.device_type)
                if third_item.ndim == 2:
                    sqi_sample = third_item  # Temporal SQI
            elif len(batch_without_id0) == 4:
                # (x, targets, sqi, sqi_sample)
                x, targets, _, sqi_sample_raw = batch_without_id0
                if not isinstance(sqi_sample_raw, torch.Tensor):
                    sqi_sample_raw = torch.from_numpy(sqi_sample_raw).float()
                sqi_sample = sqi_sample_raw.to(self.device_type)
            else:
                x, targets = batch_without_id0[0], batch_without_id0[1]
        else:
            x, targets = batch
        
        output_logits = self(x, sqi_sample=sqi_sample)
        # preds = self.predict_with_ectopic_threshold(output_logits)
        preds = torch.argmax(output_logits, dim=1)
        
        # Use the same loss function as training (with class weights support)
        if self.loss_name in ["bce", "cross_entropy", "ce"]:
            # Get per-sample loss using the configured loss function
            # Pass class_weights if available for 3-class imbalanced classification
            per_sample_loss = self.loss_fn(
                targets, output_logits, self.device_type, reduction="none", 
                class_weights=self.class_weights
            )
            loss = per_sample_loss.mean()
        else:
            raise ValueError(f"Invalid loss function: {self.loss_name}")

        # Pass logits for AUC metric, predictions for other metrics
        self.update_metrics(preds, targets, "valid", logits=output_logits)
        self.step_losses["valid"].append(loss.item())
        
        # Track per-subject data if id0 is available
        if id0_batch is not None:
            # Convert id0_batch to list if needed
            if isinstance(id0_batch, np.ndarray):
                id0_list = id0_batch.tolist()
            elif isinstance(id0_batch, torch.Tensor):
                id0_list = id0_batch.cpu().numpy().tolist()
            elif isinstance(id0_batch, (list, tuple)):
                id0_list = list(id0_batch)
            else:
                id0_list = [id0_batch]
            
            # Move predictions and targets to CPU for storage
            preds_cpu = preds.cpu().numpy()
            targets_cpu = targets.cpu().numpy()
            # Use per_sample_loss for per-subject tracking (before mean reduction)
            loss_cpu = per_sample_loss.cpu().detach().numpy()
            
            # Group by subject
            for i, subject_id in enumerate(id0_list):
                subject_id_str = str(subject_id)
                if subject_id_str not in self.valid_per_subject:
                    self.valid_per_subject[subject_id_str] = {
                        'losses': [],
                        'predictions': [],
                        'targets': []
                    }
                self.valid_per_subject[subject_id_str]['losses'].append(float(loss_cpu[i]))
                self.valid_per_subject[subject_id_str]['predictions'].append(int(preds_cpu[i]))
                self.valid_per_subject[subject_id_str]['targets'].append(int(targets_cpu[i]))

        return {"val_loss": loss.mean()}
    
    def on_validation_epoch_start(self):
        """Clear per-subject data at the start of validation epoch"""
        self.valid_per_subject.clear()
    
    def on_validation_epoch_end(self):
        """End of the validation epoch"""
        avg_loss = sum(self.step_losses["valid"]) / len(self.step_losses["valid"])

        acc = matrix = f1 = spec = auc = sensitivity = ppv = None
        acc_per_class = f1_per_class = sensitivity_per_class = ppv_per_class = None
        spec_per_class = auc_per_class = None

        if "accuracy" in self.metrics_lst:
            acc = self.metrics["metrics_" + "valid"]["accuracy"].compute()
            if self.task == "multiclass" and acc.numel() > 1:
                acc_per_class = acc

        if "cf_matrix" in self.metrics_lst:
            matrix = self.metrics["metrics_" + "valid"]["cf_matrix"].compute()

        if "f1" in self.metrics_lst:
            f1 = self.metrics["metrics_" + "valid"]["f1"].compute()
            if "f1_per_class" in self.metrics["metrics_" + "valid"]:
                f1_per_class = self.metrics["metrics_" + "valid"]["f1_per_class"].compute()
        
        if "specificity" in self.metrics_lst:
            spec = self.metrics["metrics_" + "valid"]["specificity"].compute()
            if "specificity_per_class" in self.metrics["metrics_" + "valid"]:
                spec_per_class = self.metrics["metrics_" + "valid"]["specificity_per_class"].compute()

        if "AUC"  in self.metrics_lst:
            auc = self.metrics["metrics_" + "valid"]["AUC"].compute()
            if "AUC_per_class" in self.metrics["metrics_" + "valid"]:
                auc_per_class = self.metrics["metrics_" + "valid"]["AUC_per_class"].compute()

        if "sensitivity" in self.metrics_lst:
            sensitivity = self.metrics["metrics_" + "valid"]["sensitivity"].compute()
            if "sensitivity_per_class" in self.metrics["metrics_" + "valid"]:
                sensitivity_per_class = self.metrics["metrics_" + "valid"]["sensitivity_per_class"].compute()

        if "ppv" in self.metrics_lst:
            ppv = self.metrics["metrics_" + "valid"]["ppv"].compute()
            if "ppv_per_class" in self.metrics["metrics_" + "valid"]:
                ppv_per_class = self.metrics["metrics_" + "valid"]["ppv_per_class"].compute()
        
        # Build log items
        log_items = [
            ("loss", avg_loss),
            ("accuracy", acc.mean() if acc_per_class is not None else acc),
            ("specificity", spec),
            ("AUC", auc),
            ("sensitivity", sensitivity),
            ("ppv", ppv),
            ("confusion_matrix", matrix),
            ("f1", f1),
        ]
        
        # Add per-class metrics
        if acc_per_class is not None:
            for i, acc_val in enumerate(acc_per_class):
                log_items.append((f"accuracy_class_{i}", acc_val))
        
        if f1_per_class is not None:
            for i, f1_val in enumerate(f1_per_class):
                log_items.append((f"f1_class_{i}", f1_val))
        
        if sensitivity_per_class is not None:
            for i, sens_val in enumerate(sensitivity_per_class):
                log_items.append((f"sensitivity_class_{i}", sens_val))
        
        if ppv_per_class is not None:
            for i, ppv_val in enumerate(ppv_per_class):
                log_items.append((f"ppv_class_{i}", ppv_val))
        
        if spec_per_class is not None:
            for i, spec_val in enumerate(spec_per_class):
                log_items.append((f"specificity_class_{i}", spec_val))
        
        if auc_per_class is not None:
            for i, auc_val in enumerate(auc_per_class):
                log_items.append((f"AUC_class_{i}", auc_val))
        
        self.log_all(
                items=log_items,
                phase="valid",
                prog_bar=True,
                sync_dist_group=False,
            )
        
        # Calculate per-subject macro-average metrics if per-subject data is available
        if len(self.valid_per_subject) > 0:
            # Calculate per-subject macro-average loss (each subject counts equally)
            subject_losses = []
            for subject_id, data in self.valid_per_subject.items():
                # Average loss for this subject
                subject_avg_loss = np.mean(data['losses'])
                subject_losses.append(subject_avg_loss)
            
            # Macro-average loss across subjects
            macro_avg_loss = np.mean(subject_losses) if len(subject_losses) > 0 else avg_loss
            
            # Calculate per-subject macro-average metrics
            # For each subject, calculate metrics, then average across subjects
            subject_accuracies = []
            subject_f1_scores = []
            subject_sensitivities = []
            subject_specificities = []
            subject_ppvs = []
            subject_aucs = []
            
            for subject_id, data in self.valid_per_subject.items():
                preds_subject = torch.tensor(data['predictions'], device=self.device_type)
                targets_subject = torch.tensor(data['targets'], device=self.device_type)
                
                # Calculate metrics for this subject
                if "accuracy" in self.metrics_lst:
                    acc_metric = torchmetrics.Accuracy(
                        self.task, num_classes=self.num_classes, average="none"
                    ).to(self.device_type)
                    acc_subject = acc_metric(preds_subject, targets_subject)
                    if acc_subject.numel() > 1:
                        subject_accuracies.append(acc_subject.mean().item())
                    else:
                        subject_accuracies.append(acc_subject.item())
                
                if "f1" in self.metrics_lst:
                    f1_metric = torchmetrics.F1Score(
                        self.task, num_classes=self.num_classes, average="macro"
                    ).to(self.device_type)
                    f1_subject = f1_metric(preds_subject, targets_subject)
                    subject_f1_scores.append(f1_subject.item())
                
                if "sensitivity" in self.metrics_lst:
                    if self.task == "binary":
                        sens_metric = torchmetrics.Recall(task="binary").to(self.device_type)
                    else:
                        sens_metric = torchmetrics.Recall(
                            task="multiclass", num_classes=self.num_classes, average="macro"
                        ).to(self.device_type)
                    sens_subject = sens_metric(preds_subject, targets_subject)
                    subject_sensitivities.append(sens_subject.item())
                
                if "specificity" in self.metrics_lst:
                    spec_metric = torchmetrics.Specificity(
                        self.task, num_classes=self.num_classes, average="macro"
                    ).to(self.device_type)
                    spec_subject = spec_metric(preds_subject, targets_subject)
                    subject_specificities.append(spec_subject.item())
                
                if "ppv" in self.metrics_lst:
                    if self.task == "binary":
                        ppv_metric = torchmetrics.Precision(task="binary").to(self.device_type)
                    else:
                        ppv_metric = torchmetrics.Precision(
                            task="multiclass", num_classes=self.num_classes, average="macro"
                        ).to(self.device_type)
                    ppv_subject = ppv_metric(preds_subject, targets_subject)
                    subject_ppvs.append(ppv_subject.item())
                
                # Note: AUC requires logits/probabilities, which we don't store per-subject
                # So we skip AUC for per-subject metrics
            
            # Calculate macro-averages across subjects
            per_subject_log_items = [
                ("loss_macro", macro_avg_loss),
            ]
            
            if len(subject_accuracies) > 0:
                per_subject_log_items.append(("accuracy_macro", np.mean(subject_accuracies)))
            if len(subject_f1_scores) > 0:
                per_subject_log_items.append(("f1_macro", np.mean(subject_f1_scores)))
            if len(subject_sensitivities) > 0:
                per_subject_log_items.append(("sensitivity_macro", np.mean(subject_sensitivities)))
            if len(subject_specificities) > 0:
                per_subject_log_items.append(("specificity_macro", np.mean(subject_specificities)))
            if len(subject_ppvs) > 0:
                per_subject_log_items.append(("ppv_macro", np.mean(subject_ppvs)))
            
            # Log per-subject macro-average metrics
            self.log_all(
                items=per_subject_log_items,
                phase="valid",
                prog_bar=True,
                sync_dist_group=False,
            )
            
            # Clear per-subject data for next epoch
            self.valid_per_subject.clear()
        
        self.reset_metrics("valid")
        self.step_losses["valid"].clear()
        
        # # Visualize misclassified samples if enabled and any exist
        # if self.collect_misclassified_samples and self.misclassified_samples["valid"]:
        #     self.visualize_misclassified_samples("valid")
    
    def test_step(self, batch):
        
        ecg = None
        filenames = None
        sqi_sample = None
        sqi = None

        if isinstance(batch, (list, tuple)):
            batch_list = list(batch)
            
            # Extract sqi and sqi_sample from the end (they come last)
            # Format can be: (..., sqi, sqi_sample) or (..., sqi) or (..., sqi_sample)
            if len(batch_list) >= 3:
                # Check last item
                last_item = batch_list[-1]
                if isinstance(last_item, (torch.Tensor, np.ndarray)):
                    if not isinstance(last_item, torch.Tensor):
                        last_item = torch.from_numpy(last_item).float()
                    last_item = last_item.to(self.device_type)
                    
                    if last_item.ndim == 2:
                        # Last item is sqi_sample (temporal, 2D)
                        sqi_sample = last_item
                        batch_list = batch_list[:-1]
                        
                        # Check if second-to-last is sqi (scalar, 1D)
                        if len(batch_list) >= 3:
                            second_last = batch_list[-1]
                            if isinstance(second_last, (torch.Tensor, np.ndarray)):
                                if not isinstance(second_last, torch.Tensor):
                                    second_last = torch.from_numpy(second_last).float()
                                second_last = second_last.to(self.device_type)
                                if second_last.ndim == 1:
                                    sqi = second_last
                                    batch_list = batch_list[:-1]
                    elif last_item.ndim == 1:
                        # Last item is sqi (scalar, 1D)
                        sqi = last_item
                        batch_list = batch_list[:-1]
            
            # Now process the batch without SQI
            if len(batch_list) == 4:
                # (x, targets, ecg, filenames)
                x, targets, ecg, filenames = batch_list
            elif len(batch_list) == 3:
                # Could be (x, targets, ecg) or (x, targets, filenames)
                x, targets, third_item = batch_list
                # Try to determine if third_item is ECG or filenames based on type
                if isinstance(third_item, torch.Tensor):
                    ecg = third_item
                else:
                    # Assume it's filenames (string, numpy array, etc.)
                    filenames = third_item
            elif len(batch_list) == 2:
                x, targets = batch_list
            else:
                raise ValueError(f"Unexpected batch structure received in test_step. Expected 2-4 items (after removing SQI), got {len(batch_list)}.")
        elif isinstance(batch, dict):
            x = batch.get("x") or batch.get("ppg")
            targets = batch.get("y") or batch.get("labels")
            ecg = batch.get("ecg")
            filenames = batch.get("filenames")
            sqi_data = batch.get("sqi")
            sqi_sample_data = batch.get("sqi_sample")
            if sqi_data is not None:
                if not isinstance(sqi_data, torch.Tensor):
                    sqi_data = torch.from_numpy(sqi_data).float()
                sqi = sqi_data.to(self.device_type)
            if sqi_sample_data is not None:
                if not isinstance(sqi_sample_data, torch.Tensor):
                    sqi_sample_data = torch.from_numpy(sqi_sample_data).float()
                sqi_sample = sqi_sample_data.to(self.device_type)
        else:
            raise ValueError("Unsupported batch type received in test_step.")

        output_logits = self(x, sqi_sample=sqi_sample)
        # preds = self.predict_with_ectopic_threshold(output_logits)
        preds = torch.argmax(output_logits, dim=1)
        loss = F.cross_entropy(output_logits, targets)

        # Pass logits for AUC metric, predictions for other metrics
        self.update_metrics(preds, targets, "test", logits=output_logits)
        self.step_losses["test"].append(loss)
        
        # Store predictions, targets, logits, and SQI for stratified metrics
        self.test_predictions.append(preds.detach().cpu())
        self.test_targets.append(targets.detach().cpu())
        self.test_logits.append(output_logits.detach().cpu())
        if sqi is not None:
            self.test_sqi.append(sqi.detach().cpu())
        else:
            # If SQI not available, create dummy values (will skip stratified metrics)
            self.test_sqi.append(None)
        
        # Store embeddings before and after attention
        self.test_z_backbone.append(self.model.feat_extracter.z_backbone.detach().cpu())
        self.test_z_att.append(self.model.feat_extracter.z_att.detach().cpu())


        # Collect misclassified samples if enabled (collect ALL, no limit)
        if self.collect_misclassified_samples:
            misclassified_mask = (preds != targets)
            if misclassified_mask.any():
                misclassified_indices = torch.where(misclassified_mask)[0]
                for idx in misclassified_indices:
                    sample_payload = {
                        'input': x[idx].detach().cpu().numpy(),
                        'prediction': preds[idx].item(),
                        'true_label': targets[idx].item(),
                        'probabilities': F.softmax(output_logits[idx], dim=0).detach().cpu().numpy()
                    }

                    if ecg is not None:
                        sample_payload['ecg'] = ecg[idx].detach().cpu().numpy()
                    
                    # Store filename if available
                    if filenames is not None:
                        try:
                            if isinstance(filenames, (list, tuple)):
                                sample_payload['filename'] = str(filenames[idx])
                            elif isinstance(filenames, np.ndarray):
                                # Handle numpy array of strings
                                filename_val = filenames[idx]
                                sample_payload['filename'] = str(filename_val) if not isinstance(filename_val, np.ndarray) else str(filename_val.item())
                            elif isinstance(filenames, torch.Tensor):
                                sample_payload['filename'] = str(filenames[idx].item()) if filenames[idx].numel() == 1 else str(filenames[idx].cpu().numpy())
                            else:
                                sample_payload['filename'] = str(filenames[idx])
                        except (IndexError, TypeError) as e:
                            # If we can't extract the filename, skip it
                            print(f"Warning: Could not extract filename for misclassified sample: {e}", flush=True)

                    self.misclassified_samples["test"].append(sample_payload)

        return {'test_loss': loss}

    def on_test_epoch_start(self):
        """Clear misclassified samples and test data at the start of test epoch"""
        if self.collect_misclassified_samples:
            self.misclassified_samples["test"].clear()
        # Clear stored test data for stratified metrics
        self.test_predictions.clear()
        self.test_targets.clear()
        self.test_logits.clear()
        self.test_sqi.clear()
        self.test_z_backbone.clear()
        self.test_z_att.clear()
    
    def on_test_epoch_end(self):
        """End of the test epoch"""
        avg_loss = sum(self.step_losses["test"]) / len(self.step_losses["test"])

        acc = matrix = f1 = spec = auc = sensitivity = ppv = None
        acc_per_class = f1_per_class = sensitivity_per_class = ppv_per_class = None
        spec_per_class = auc_per_class = None

        if "accuracy" in self.metrics_lst:
            acc = self.metrics["metrics_" + "test"]["accuracy"].compute()
            if self.task == "multiclass" and acc.numel() > 1:
                acc_per_class = acc

        if "cf_matrix" in self.metrics_lst:
            matrix = self.metrics["metrics_" + "test"]["cf_matrix"].compute()

        if "f1" in self.metrics_lst:
            f1 = self.metrics["metrics_" + "test"]["f1"].compute()
            if "f1_per_class" in self.metrics["metrics_" + "test"]:
                f1_per_class = self.metrics["metrics_" + "test"]["f1_per_class"].compute()
        
        if "specificity" in self.metrics_lst:
            spec = self.metrics["metrics_" + "test"]["specificity"].compute()
            if "specificity_per_class" in self.metrics["metrics_" + "test"]:
                spec_per_class = self.metrics["metrics_" + "test"]["specificity_per_class"].compute()

        if "AUC"  in self.metrics_lst:
            auc = self.metrics["metrics_" + "test"]["AUC"].compute()
            if "AUC_per_class" in self.metrics["metrics_" + "test"]:
                auc_per_class = self.metrics["metrics_" + "test"]["AUC_per_class"].compute()

        if "sensitivity" in self.metrics_lst:
            sensitivity = self.metrics["metrics_" + "test"]["sensitivity"].compute()
            if "sensitivity_per_class" in self.metrics["metrics_" + "test"]:
                sensitivity_per_class = self.metrics["metrics_" + "test"]["sensitivity_per_class"].compute()

        if "ppv" in self.metrics_lst:
            ppv = self.metrics["metrics_" + "test"]["ppv"].compute()
            if "ppv_per_class" in self.metrics["metrics_" + "test"]:
                ppv_per_class = self.metrics["metrics_" + "test"]["ppv_per_class"].compute()
        
        # Build log items
        log_items = [
            ("loss", avg_loss),
            ("accuracy", acc.mean() if acc_per_class is not None else acc),
            ("specificity", spec),
            ("AUC", auc),
            ("sensitivity", sensitivity),
            ("ppv", ppv),
            ("confusion_matrix", matrix),
            ("f1", f1),
        ]
        
        # Add per-class metrics
        if acc_per_class is not None:
            for i, acc_val in enumerate(acc_per_class):
                log_items.append((f"accuracy_class_{i}", acc_val))
        
        if f1_per_class is not None:
            for i, f1_val in enumerate(f1_per_class):
                log_items.append((f"f1_class_{i}", f1_val))
        
        if sensitivity_per_class is not None:
            for i, sens_val in enumerate(sensitivity_per_class):
                log_items.append((f"sensitivity_class_{i}", sens_val))
        
        if ppv_per_class is not None:
            for i, ppv_val in enumerate(ppv_per_class):
                log_items.append((f"ppv_class_{i}", ppv_val))
        
        if spec_per_class is not None:
            for i, spec_val in enumerate(spec_per_class):
                log_items.append((f"specificity_class_{i}", spec_val))
        
        if auc_per_class is not None:
            for i, auc_val in enumerate(auc_per_class):
                log_items.append((f"AUC_class_{i}", auc_val))
        
        self.log_all(
                items=log_items,
                phase="test",
                prog_bar=True,
                sync_dist_group=False,
            )
        
        self.reset_metrics("test")
        self.step_losses["test"].clear()
        
        # Compute SQI-stratified metrics if SQI data is available
        if self.test_sqi and any(sqi is not None for sqi in self.test_sqi):
            self._compute_sqi_stratified_metrics()
        
        # Save and report test confusion matrix
        if "cf_matrix" in self.metrics_lst and matrix is not None:
            self.save_test_confusion_matrix(matrix)
        
        # Visualize misclassified samples if enabled and any exist
        if self.collect_misclassified_samples and self.misclassified_samples["test"]:
            self.visualize_misclassified_samples("test")
            # Save filenames of misclassified samples
            self.save_misclassified_filenames("test")
        
        # Save embeddings with labels and SQI
        self.save_embeddings_with_labels_sqi()

    def predict_step(self, batch):
        x, targets = batch
        output_logits = self(x)
        # preds = self.predict_with_ectopic_threshold(output_logits)
        preds = torch.argmax(output_logits, dim=1)
        return preds


    
    













    

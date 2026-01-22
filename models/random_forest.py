from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    accuracy_score, f1_score, recall_score, precision_score,
    roc_curve, auc
)
import matplotlib.pyplot as plt


class RandomForest():
    def __init__(self, n_feats_to_select, n_estimators=100, max_depth=None, 
                 min_samples_split=2, min_samples_leaf=1, 
                 class_weight='balanced', random_state=42,
                 get_summary=True, confusion_matrix=True, 
                 classification_report=True, aus_score=True, plot_roc=True):
        """
        Initialize Random Forest model.
        
        Args:
            n_feats_to_select: Number of features to select using RFE
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the tree (None for no limit)
            min_samples_split: Minimum number of samples required to split a node
            min_samples_leaf: Minimum number of samples required at a leaf node
            class_weight: Class weight balancing ('balanced', 'balanced_subsample', or None)
            random_state: Random seed for reproducibility
            get_summary: Whether to print model summary
            confusion_matrix: Whether to compute and print confusion matrix
            classification_report: Whether to print classification report
            aus_score: Whether to compute ROC AUC score
            plot_roc: Whether to plot ROC curve
        """
        self.n_feats_to_select = n_feats_to_select
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state
        self.get_summary = get_summary
        self.confusion_matrix = confusion_matrix
        self.classification_report = classification_report
        self.aus_score = aus_score
        self.plot_roc = plot_roc
        
        # Initialize Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
        self.estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.selected_feats = None
        self.os_train_x = None

    def _select_features(self, x: pd.DataFrame, y: pd.Series):
        """Select features using RFE."""
        rfe = RFE(estimator=self.estimator, n_features_to_select=self.n_feats_to_select)
        rfe = rfe.fit(x, y.values.ravel())
        selected_feats = x.columns[rfe.support_]
        
        print(f'Features selected based on RFE: {selected_feats.tolist()}')
        return selected_feats
    
    def train(self, train_x: pd.DataFrame, train_y: pd.Series):
        """Train the Random Forest model."""
        # Select features
        self.selected_feats = self._select_features(train_x, train_y)
        self.os_train_x = train_x[self.selected_feats]
        
        # Train model
        self.model.fit(self.os_train_x, train_y)
        
        if self.get_summary:
            print(f"\nRandom Forest Model Summary:")
            print(f"  Number of estimators: {self.n_estimators}")
            print(f"  Max depth: {self.max_depth}")
            print(f"  Min samples split: {self.min_samples_split}")
            print(f"  Min samples leaf: {self.min_samples_leaf}")
            print(f"  Class weight: {self.class_weight}")
            print(f"  Selected features: {self.selected_feats.tolist()}")
            print(f"  Training samples: {len(train_y)}")
            print(f"  Number of classes: {len(train_y.unique())}")
    
    def test(self, test_x: pd.DataFrame, test_y: pd.Series):
        """Test the Random Forest model and compute metrics."""
        os_test_x = test_x[self.selected_feats]
        
        # Get predictions
        pred_y = self.model.predict(os_test_x)
        pred_y_prob = self.model.predict_proba(os_test_x)
        
        # Compute accuracy
        accuracy = accuracy_score(test_y, pred_y)
        print(f'\nAccuracy of Random Forest classifier on test set: {accuracy:.4f}')
        
        # Get number of classes
        num_classes = len(np.unique(test_y))
        
        # Compute metrics
        if self.confusion_matrix:
            matrix = self._confusion_matrix(test_y, pred_y, num_classes)
        
        if self.classification_report:
            report = self._classification_report(test_y, pred_y)
        
        # Compute comprehensive metrics (same as ResNet)
        metrics = self._compute_all_metrics(test_y, pred_y, pred_y_prob, num_classes)
        
        if self.aus_score:
            self._compute_roc_auc_score(test_y, pred_y_prob, num_classes)
        
        if self.plot_roc:
            self._plot_roc(test_y, pred_y_prob, num_classes)
        
        # Store metrics for retrieval
        self.test_metrics = metrics
        return metrics

    def _confusion_matrix(self, test_y, pred_y, num_classes):
        """Compute and print confusion matrix."""
        matrix = confusion_matrix(test_y, pred_y)
        print(f'\nConfusion Matrix:')
        print(matrix)
        return matrix
    
    def _classification_report(self, test_y, pred_y):
        """Compute and print classification report."""
        report = classification_report(test_y, pred_y)
        print(f'\nClassification Report:')
        print(report)
        return report
    
    def _compute_all_metrics(self, test_y, pred_y, pred_y_prob, num_classes):
        """Compute all metrics similar to ResNet evaluation."""
        print(f'\n{"="*60}')
        print('Comprehensive Metrics (ResNet-style):')
        print(f'{"="*60}')
        
        # Accuracy (macro and per-class)
        accuracy_macro = accuracy_score(test_y, pred_y)
        print(f'Accuracy (macro): {accuracy_macro:.4f}')
        
        # Per-class accuracy (one-vs-rest)
        accuracy_per_class = []
        for i in range(num_classes):
            y_binary = (test_y == i).astype(int)
            pred_binary = (pred_y == i).astype(int)
            acc_class = accuracy_score(y_binary, pred_binary)
            accuracy_per_class.append(acc_class)
            print(f'  Accuracy class_{i}: {acc_class:.4f}')
        
        # F1 score (macro and per-class)
        f1_macro = f1_score(test_y, pred_y, average='macro')
        f1_per_class = f1_score(test_y, pred_y, average=None)
        print(f'\nF1 Score (macro): {f1_macro:.4f}')
        for i, f1_val in enumerate(f1_per_class):
            print(f'  F1 class_{i}: {f1_val:.4f}')
        
        # Sensitivity/Recall (macro and per-class)
        sensitivity_macro = recall_score(test_y, pred_y, average='macro', zero_division=0)
        sensitivity_per_class = recall_score(test_y, pred_y, average=None, zero_division=0)
        print(f'\nSensitivity/Recall (macro): {sensitivity_macro:.4f}')
        for i, sens_val in enumerate(sensitivity_per_class):
            print(f'  Sensitivity class_{i}: {sens_val:.4f}')
        
        # PPV/Precision (macro and per-class)
        ppv_macro = precision_score(test_y, pred_y, average='macro', zero_division=0)
        ppv_per_class = precision_score(test_y, pred_y, average=None, zero_division=0)
        print(f'\nPPV/Precision (macro): {ppv_macro:.4f}')
        for i, ppv_val in enumerate(ppv_per_class):
            print(f'  PPV class_{i}: {ppv_val:.4f}')
        
        # Specificity (macro and per-class) - computed from confusion matrix
        cm = confusion_matrix(test_y, pred_y)
        specificity_per_class = []
        for i in range(num_classes):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            spec_class = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificity_per_class.append(spec_class)
        specificity_macro = np.mean(specificity_per_class)
        print(f'\nSpecificity (macro): {specificity_macro:.4f}')
        for i, spec_val in enumerate(specificity_per_class):
            print(f'  Specificity class_{i}: {spec_val:.4f}')
        
        # AUC (macro and per-class) - one-vs-rest
        if num_classes == 2:
            # Binary classification
            auc_macro = roc_auc_score(test_y, pred_y_prob[:, 1])
            auc_per_class = [auc_macro, auc_macro]  # Same for binary
        else:
            # Multiclass - one-vs-rest
            auc_per_class = []
            for i in range(num_classes):
                y_binary = (test_y == i).astype(int)
                if len(np.unique(y_binary)) > 1:  # Check if both classes present
                    auc_class = roc_auc_score(y_binary, pred_y_prob[:, i])
                else:
                    auc_class = 0.0
                auc_per_class.append(auc_class)
            auc_macro = np.mean(auc_per_class)
        
        print(f'\nAUC (macro): {auc_macro:.4f}')
        for i, auc_val in enumerate(auc_per_class):
            print(f'  AUC class_{i}: {auc_val:.4f}')
        
        print(f'{"="*60}\n')
        
        # Return metrics dictionary
        metrics = {
            'test_accuracy': accuracy_macro,
            'test_f1': f1_macro,
            'test_sensitivity': sensitivity_macro,
            'test_ppv': ppv_macro,
            'test_specificity': specificity_macro,
            'test_AUC': auc_macro,
        }
        
        # Add per-class metrics
        for i in range(num_classes):
            metrics[f'test_accuracy_class_{i}'] = accuracy_per_class[i]
            metrics[f'test_f1_class_{i}'] = f1_per_class[i]
            metrics[f'test_sensitivity_class_{i}'] = sensitivity_per_class[i]
            metrics[f'test_ppv_class_{i}'] = ppv_per_class[i]
            metrics[f'test_specificity_class_{i}'] = specificity_per_class[i]
            metrics[f'test_AUC_class_{i}'] = auc_per_class[i]
        
        return metrics
    
    def _compute_roc_auc_score(self, test_y, pred_y_prob, num_classes):
        """Compute ROC AUC score."""
        if num_classes == 2:
            auc_score = roc_auc_score(test_y, pred_y_prob[:, 1])
            print(f'\nROC AUC Score (binary): {auc_score:.4f}')
        else:
            # Multiclass - one-vs-rest
            auc_scores = []
            for i in range(num_classes):
                y_binary = (test_y == i).astype(int)
                if len(np.unique(y_binary)) > 1:
                    auc_class = roc_auc_score(y_binary, pred_y_prob[:, i])
                    auc_scores.append(auc_class)
                else:
                    auc_scores.append(0.0)
            auc_score = np.mean(auc_scores)
            print(f'\nROC AUC Score (macro, one-vs-rest): {auc_score:.4f}')
            for i, auc_val in enumerate(auc_scores):
                print(f'  ROC AUC class_{i}: {auc_val:.4f}')
   
    def _plot_roc(self, test_y, pred_y_prob, num_classes):
        """Plot ROC curve(s)."""
        plt.figure(figsize=(10, 8))
        
        if num_classes == 2:
            # Binary classification - single ROC curve
            fpr, tpr, _ = roc_curve(test_y, pred_y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.4f})', linewidth=2)
        else:
            # Multiclass - one-vs-rest ROC curves
            for i in range(num_classes):
                y_binary = (test_y == i).astype(int)
                if len(np.unique(y_binary)) > 1:
                    fpr, tpr, _ = roc_curve(y_binary, pred_y_prob[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.4f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'r--', label='Random', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.savefig('RandomForest_ROC.png', dpi=300, bbox_inches='tight')
        print('\nROC curve saved as RandomForest_ROC.png')
        plt.close()


# Metrics Explanation for 3-Class Classification

This document explains how sensitivity, specificity, PPV (Positive Predictive Value), accuracy, and AUC are calculated for each class and overall in the 3-class classification system (Normal, Ectopic, VT).

## Understanding the Metrics

### Per-Class Metrics (average="none")

For each class, metrics are calculated by treating that class as the "positive" class and all other classes as "negative". This is called a **one-vs-rest** approach.

#### For a 3-class problem (Normal=0, Ectopic=1, VT=2):

**Example Confusion Matrix:**
```
                Predicted
              Normal  Ectopic  VT
Actual Normal    TP      FN    FN
      Ectopic    FN      TP    FN
      VT         FN      FN    TP
```

For each class, we calculate:
- **TP (True Positives)**: Correctly predicted as that class
- **TN (True Negatives)**: Correctly predicted as NOT that class
- **FP (False Positives)**: Incorrectly predicted as that class
- **FN (False Negatives)**: Actually that class but predicted as something else

### 1. Per-Class Sensitivity (Recall)

**Definition**: The proportion of actual positives (for a class) that are correctly identified.

**Formula for class i**: 
```
Sensitivity_i = TP_i / (TP_i + FN_i)
```

**What it means**:
- **Sensitivity_class_0 (Normal)**: Of all samples that are actually Normal, what percentage did we correctly identify?
- **Sensitivity_class_1 (Ectopic)**: Of all samples that are actually Ectopic, what percentage did we correctly identify?
- **Sensitivity_class_2 (VT)**: Of all samples that are actually VT, what percentage did we correctly identify?

**Interpretation**: 
- High sensitivity = good at detecting that class (few false negatives)
- Low sensitivity = missing many samples of that class

### 2. Per-Class Specificity

**Definition**: The proportion of actual negatives (NOT that class) that are correctly identified as negative.

**Formula for class i**:
```
Specificity_i = TN_i / (TN_i + FP_i)
```

**What it means**:
- **Specificity_class_0 (Normal)**: Of all samples that are NOT Normal, what percentage did we correctly identify as not Normal?
- **Specificity_class_1 (Ectopic)**: Of all samples that are NOT Ectopic, what percentage did we correctly identify as not Ectopic?
- **Specificity_class_2 (VT)**: Of all samples that are NOT VT, what percentage did we correctly identify as not VT?

**Interpretation**:
- High specificity = good at avoiding false alarms for that class
- Low specificity = incorrectly predicting that class when it's not present

### 3. Per-Class PPV (Positive Predictive Value / Precision)

**Definition**: The proportion of predicted positives (for a class) that are actually positive.

**Formula for class i**:
```
PPV_i = TP_i / (TP_i + FP_i)
```

**What it means**:
- **PPV_class_0 (Normal)**: When we predict Normal, how often are we correct?
- **PPV_class_1 (Ectopic)**: When we predict Ectopic, how often are we correct?
- **PPV_class_2 (VT)**: When we predict VT, how often are we correct?

**Interpretation**:
- High PPV = when we predict that class, we're usually right
- Low PPV = many false positives for that class

### 4. Per-Class Accuracy

**Definition**: The proportion of correct predictions for that class (both positive and negative).

**Formula for class i**:
```
Accuracy_i = (TP_i + TN_i) / (TP_i + TN_i + FP_i + FN_i)
```

**What it means**:
- **Accuracy_class_0**: Overall correctness when considering Normal vs. not Normal
- **Accuracy_class_1**: Overall correctness when considering Ectopic vs. not Ectopic
- **Accuracy_class_2**: Overall correctness when considering VT vs. not VT

**Note**: In the current implementation, per-class accuracy is calculated using `average="none"` which gives the accuracy for each class in the one-vs-rest setting.

### 5. Per-Class AUC (Area Under ROC Curve)

**Definition**: The area under the ROC curve for that class in a one-vs-rest setting.

**What it means**:
- **AUC_class_0**: How well can we distinguish Normal from (Ectopic + VT)?
- **AUC_class_1**: How well can we distinguish Ectopic from (Normal + VT)?
- **AUC_class_2**: How well can we distinguish VT from (Normal + Ectopic)?

**Interpretation**:
- AUC = 1.0: Perfect separation
- AUC = 0.5: Random guessing
- AUC > 0.7: Generally considered acceptable
- AUC > 0.9: Excellent discrimination

## Overall Metrics (Macro-Averaged)

All overall metrics use **macro-averaging**, which means:
1. Calculate the metric for each class separately
2. Take the average across all classes
3. Each class contributes equally regardless of class size

### Formula for Macro-Averaging:
```
Macro_Metric = (Metric_class_0 + Metric_class_1 + Metric_class_2) / 3
```

### 1. Overall Sensitivity (Macro-Averaged Recall)
```
Overall_Sensitivity = (Sensitivity_0 + Sensitivity_1 + Sensitivity_2) / 3
```
- Average sensitivity across all three classes
- Each class weighted equally

### 2. Overall Specificity (Macro-Averaged)
```
Overall_Specificity = (Specificity_0 + Specificity_1 + Specificity_2) / 3
```
- Average specificity across all three classes
- Each class weighted equally

### 3. Overall PPV (Macro-Averaged Precision)
```
Overall_PPV = (PPV_0 + PPV_1 + PPV_2) / 3
```
- Average precision across all three classes
- Each class weighted equally

### 4. Overall Accuracy
```
Overall_Accuracy = (Accuracy_0 + Accuracy_1 + Accuracy_2) / 3
```
- In the current implementation, this is the mean of per-class accuracies
- Note: This is different from overall classification accuracy (correct predictions / total predictions)

### 5. Overall AUC (Macro-Averaged AUROC)
```
Overall_AUC = (AUC_0 + AUC_1 + AUC_2) / 3
```
- Average AUC across all three classes
- Each class weighted equally

## Macro vs. Micro Averaging

### Macro-Averaging (Current Implementation)
- **How**: Calculate metric for each class, then average
- **Weight**: Each class contributes equally
- **Use case**: When you want to treat all classes equally, regardless of class imbalance
- **Advantage**: Not biased by class size
- **Disadvantage**: May be misleading if classes are very imbalanced

### Micro-Averaging (Not Currently Used)
- **How**: Aggregate all TP, TN, FP, FN across classes, then calculate metric
- **Weight**: Larger classes contribute more
- **Use case**: When you want overall performance weighted by class frequency
- **Advantage**: Reflects overall performance on the dataset
- **Disadvantage**: Dominated by majority class

### Example Comparison:

**Scenario**: 
- Normal: 1000 samples, Sensitivity = 0.9
- Ectopic: 100 samples, Sensitivity = 0.7
- VT: 50 samples, Sensitivity = 0.8

**Macro-Averaged Sensitivity**: (0.9 + 0.7 + 0.8) / 3 = 0.8
- Each class weighted equally

**Micro-Averaged Sensitivity**: 
- TP_total = (900 + 70 + 40) = 1010
- (TP + FN)_total = (1000 + 100 + 50) = 1150
- Micro = 1010 / 1150 = 0.878
- Weighted by class size (Normal dominates)

## Current Implementation Summary

| Metric | Overall | Per-Class | Averaging Method |
|--------|---------|-----------|------------------|
| **Accuracy** | Macro (mean of per-class) | One-vs-rest | Macro |
| **Sensitivity** | Macro | One-vs-rest | Macro |
| **Specificity** | Macro | One-vs-rest | Macro |
| **PPV** | Macro | One-vs-rest | Macro |
| **AUC** | Macro | One-vs-rest | Macro |
| **F1** | Macro | One-vs-rest | Macro |

## Why Macro-Averaging?

Macro-averaging is appropriate for imbalanced datasets because:
1. **Equal importance**: All classes (Normal, Ectopic, VT) are clinically important
2. **Fair evaluation**: Minority classes (Ectopic, VT) get equal weight
3. **Class-agnostic**: Performance is evaluated per-class, not dominated by majority class

## Clinical Interpretation

For a 3-class cardiac rhythm classification:

- **High Sensitivity for VT**: Critical - we don't want to miss VT cases
- **High Specificity for Normal**: Important - avoid false alarms
- **Balanced PPV**: Ensures predictions are reliable across all classes
- **High AUC**: Indicates good discrimination ability for each class

## Logged Metrics in TensorBoard/CSV

### Overall Metrics:
- `train_accuracy`, `train_sensitivity`, `train_specificity`, `train_ppv`, `train_AUC`

### Per-Class Metrics:
- `train_accuracy_class_0`, `train_accuracy_class_1`, `train_accuracy_class_2`
- `train_sensitivity_class_0`, `train_sensitivity_class_1`, `train_sensitivity_class_2`
- `train_specificity_class_0`, `train_specificity_class_1`, `train_specificity_class_2`
- `train_ppv_class_0`, `train_ppv_class_1`, `train_ppv_class_2`
- `train_AUC_class_0`, `train_AUC_class_1`, `train_AUC_class_2`

Same pattern for `valid_*` and `test_*` metrics.


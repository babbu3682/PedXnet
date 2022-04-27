
from monai.metrics import ROCAUCMetric, DiceMetric, ConfusionMatrixMetric 
from monai.transforms import AsDiscrete, Activations
from monai.metrics.utils import MetricReduction



## Metric 
dice_metric    = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=False)
dice_metric1   = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=False)
confuse_metric = ConfusionMatrixMetric(include_background=True, metric_name=["f1 score", "accuracy", "sensitivity", "specificity",], compute_sample=False, reduction=MetricReduction.MEAN, get_not_nans=False)
auc_metric     = ROCAUCMetric() # Average Macro



# Post-processing define
Pred_To_16_Onehot  = AsDiscrete(argmax=True,  to_onehot=True, num_classes=16, threshold=0.5, rounding=None)
Label_To_16_Onehot = AsDiscrete(argmax=False, to_onehot=True, num_classes=16, threshold=0.5, rounding=None)
Softmax_To_Prob    = Activations(softmax=True)
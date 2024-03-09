from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import numpy as np

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def sensitivity(y_true, y_pred):
    return recall_score(y_true, y_pred)

def f1_metric(y_true, y_pred):
    return f1_score(y_true, y_pred)

def specificity(y_true, y_pred):
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    return true_negative / (true_negative + false_positive)

def calculate_ppv_npv(y_true, y_pred):
    """
    PPV와 NPV를 계산합니다.

    Parameters:
    y_true (list or array): 실제 타겟 값
    y_pred (list or array): 예측된 타겟 값

    Returns:
    tuple: (PPV, NPV)
    """
    # 혼동 행렬(confusion matrix) 계산
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # PPV와 NPV 계산
    ppv = tp / (tp + fp) if (tp + fp) else 0  # 양성 예측 가치
    npv = tn / (tn + fn) if (tn + fn) else 0  # 음성 예측 가치

    return ppv, npv

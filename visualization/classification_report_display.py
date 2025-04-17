from sklearn.metrics import classification_report

def get_classification_report_df(y_true, y_pred):
    report = classification_report(y_true, y_pred)
    return report
import os
import random
import torch
import numpy as np





def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



class EarlyStopping():
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, model_save_path, patience=5, verbose=True, delta=0):
        """
        Args:
            model_save_path (str): model saved at.
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.model_save_path = model_save_path
        self.patience = patience
        self.patience_counter = 0
        self.hist_dev_scores = []
        self.verbose = verbose
        self.delta = delta

    def __call__(self, dev_score, model):

        is_better = len(self.hist_dev_scores) == 0 or dev_score >= max(self.hist_dev_scores) + self.delta
        self.hist_dev_scores.append(dev_score)

        if is_better:
            self.patience_counter = 0
            self.save_checkpoint(dev_score, model)

        else:
            self.patience_counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.patience_counter} out of {self.patience}")
            if self.patience_counter >= self.patience:
                self.patience_counter = 0
                return True

        return False


    def save_checkpoint(self, dev_score, model):
        '''
        Saving model when validation loss increasing.
        '''
        if self.verbose:
            prev_best_dev_score = max(self.hist_dev_scores[:-1]) if len(self.hist_dev_scores[:-1])>0 else -np.inf
            print(f"Validation score increased ({prev_best_dev_score:.4f} --> {dev_score:.4f}). Save currently best model at {self.model_save_path}")
        model.save(self.model_save_path)
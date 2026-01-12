import torch
import torch.nn as nn

class LogScaledHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        """
        Initialize the LogScaledHuberLoss class.

        Parameters:
        - delta (float): The threshold at which to switch from squared error to linear error.
        """
        super(LogScaledHuberLoss, self).__init__()
        self.delta = delta

    def forward(self, predictions, targets):
        """
        Compute the log-scaled Huber loss between predictions and targets.

        Parameters:
        - predictions (torch.Tensor): The predicted values.
        - targets (torch.Tensor): The ground truth values.

        Returns:
        - torch.Tensor: The computed Log-Scaled Huber Loss.
        """

        print(f"pred[1]: {predictions[1]}\t actual[1]: {targets[1]}\n")

        # Apply log-scaling (adding 1 to avoid log(0) issues)
        log_preds = torch.log(predictions + 1)
        log_targets = torch.log(targets + 1)

        # Calculate the absolute differences
        abs_diff = torch.abs(log_preds - log_targets)

        # Apply Huber Loss formula with delta threshold
        loss = torch.where(abs_diff <= self.delta,
                           0.5 * abs_diff ** 2,                # Squared error for small differences
                           self.delta * abs_diff - 0.5 * self.delta ** 2)  # Linear error for large differences

        # Return the mean loss
        return loss.mean()


class SQ_MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = 2.0
        self.delta = 20.0

    def assign_penalization(self, actual):
        penalized = torch.zeros(len(actual), dtype=torch.bool, device=actual.device)
        for i, iat in enumerate(actual):
            if iat >= self.delta:
                penalized[i] = True
        return penalized

    def forward(self, pred, actual):
        penalized = self.assign_penalization(actual)
        print(f"pred[1]: {pred[1]}\t actual[1]: {actual[1]}\n")
        p_preds, p_iats = pred[penalized], actual[penalized]
        msle_preds, msle_iats = pred[~penalized], actual[~penalized]

        # Compute losses
        if len(msle_preds) > 0:
            msle_loss = self.mse(torch.log(msle_preds + 1), torch.log(msle_iats + 1))
        else:
            msle_loss = 0.0

        if len(p_preds) > 0:
            p_msle_loss = self.mse(torch.log(p_preds + 1), torch.log(p_iats + 1))
        else:
            p_msle_loss = 0.0

        if msle_loss == 0.0 or p_msle_loss == 0.0:
            loss = msle_loss + p_msle_loss
        else:
            loss = (msle_loss + p_msle_loss) / 2.0
        return loss

class SQ_MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = 0.5
        self.delta = 100.0

    def assign_penalization(self, actual):
        penalized = torch.zeros(len(actual), dtype=torch.bool, device=actual.device)
        for i, iat in enumerate(actual):
            if iat >= self.delta:
                penalized[i] = True
        return penalized

    def forward(self, pred, actual):
        penalized = self.assign_penalization(actual)
        print(f"pred[1]: {pred[1]}\t actual[1]: {actual[1]}\n")
        p_preds, p_iats = pred[penalized], actual[penalized]
        mse_preds, mse_iats = pred[~penalized], actual[~penalized]

        # Compute losses
        if len(mse_preds) > 0:
            mse_loss = self.mse(mse_preds, mse_iats)
        else:
            mse_loss = 0.0

        if len(p_preds) > 0:
            p_mse_loss = self.alpha * self.mse(p_preds, p_iats)
        else:
            p_mse_loss = 0.0

        if mse_loss == 0.0 or p_mse_loss == 0.0:
            loss = mse_loss + p_mse_loss
        else:
            loss = (mse_loss + p_mse_loss) / 2.0
        return loss

class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        print(f"pred[1]: {pred[1]}\t actual[1]: {actual[1]}\n")
        loss = self.mse(torch.log(pred + 1), torch.log(actual + 1))
        return loss

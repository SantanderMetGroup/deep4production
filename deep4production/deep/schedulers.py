# Custom d4d schedulers. Set the following lines in the .yaml file:
# model_info: 
#     training_params:
#         scheduler_params:
#             module: torch.optim.lr_scheduler
#             name: LambdaLR
#             # kwargs of scheduler selected:
#             lr_lambda: rampup_expdecay  # function from deep4production.deep.schedulers
#             lr_lambda_kwargs:
#               rampup_epochs: 5
#               lr_decay: 0.999
#               decay_interval: 50

# Info here: https://github.com/NVIDIA/physicsnemo/blob/main/examples/weather/corrdiff/conf/base/training/base_all.yaml
def rampup_expdecay(
    step: int,
    base_lr: float = 1e-3,
    rampup_steps: int = 1000,
    decay_rate: float = 0.999,
    decay_interval: int = 50,
    terminal_value: float = 1e-6
) -> float:
    """
    Lambda function for linear ramp-up + continuous exponential decay.
    Compatible with torch.optim.lr_scheduler.LambdaLR.

    Parameters
    ----------
    step : int
        Current training step or epoch (PyTorch passes this automatically).
    base_lr : float
        Initial learning rate (maximum before decay).
    rampup_steps : int
        Number of steps for linear warm-up.
    decay_rate : float
        Exponential decay rate (per `decay_interval`).
    decay_interval : int
        Interval of steps controlling decay speed.
    terminal_value : float
        Minimum LR (relative to base_lr), prevents LR from going to 0.

    Returns
    -------
    float
        Multiplier for the base learning rate.
    """
    # --- Linear ramp-up ---
    if step < rampup_steps:
        lr = base_lr * (step / rampup_steps)
    else:
        # --- Continuous exponential decay ---
        decay_progress = (step - rampup_steps) / max(1, decay_interval)
        lr = base_lr * (decay_rate ** decay_progress)
        # --- Apply terminal floor ---
        lr = max(lr, float(terminal_value))
    # LambdaLR expects a *multiplier* relative to base_lr:
    return lr / base_lr

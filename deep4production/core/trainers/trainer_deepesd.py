## Load libraries
import torch

## Deep4production
from deep4production.core.trainers.trainer import trainer


##################################################################################################################################
class trainer_custom(trainer):
    """
    Custom trainer class for DeepESD models using ensemble members.
    Purpose: Handles batch training, ensemble prediction, and loss computation for CNN-based models.
    Parameters:
        data (dict): Dataset configuration.
        dataloader (dict): Dataloader parameters.
        id_dir (str): Experiment directory.
        model_info (dict): Model, loss, saving, and training parameters.
        graph (dict): Graph configuration (optional).
        d4dpy (dict): Custom pydataset configuration.
        Mlflow (dict): MLflow tracking configuration.
    """

    def __init__(
        self,
        data,
        dataloader,
        id_dir,
        model_info,
        graph,
        d4dpy,
        Mlflow,
        normalizer_info_x=None,
        normalizer_info_y=None,
        normalizer_info_f=None,
        hardware=None,
    ):
        ######### Call parent constructor to initialize common attributes #########
        super().__init__(
            data=data,
            dataloader=dataloader,
            id_dir=id_dir,
            model_info=model_info,
            graph=graph,
            d4dpy=d4dpy,
            Mlflow=Mlflow,
            normalizer_info_x=normalizer_info_x,
            normalizer_info_y=normalizer_info_y,
            normalizer_info_f=normalizer_info_f,
            hardware=hardware,
        )

    # -------------------------------------------------------------------------
    def model_backprop(
        self,
        model,
        data,
        optimizer,
        loss_function,
        device,
        is_this_training=True,
        members=2,
    ):
        """
        Performs a single forward and backward pass for a batch using ensemble prediction.
        Purpose: Runs forward passes for each ensemble member, stacks predictions, computes loss, and performs backpropagation.
        Parameters:
            model: PyTorch model.
            data: Tuple of input, target, and forcing arrays.
            optimizer: PyTorch optimizer.
            loss_function: Loss function callable.
            device: Device string ('cpu' or 'cuda').
            is_this_training (bool): Whether to perform backpropagation.
            members (int): Number of ensemble members.
        Returns:
            torch.Tensor: Detached loss tensor for the batch.
        """
        # --- Get arrays ---
        x, y, f = data
        non_blocking = self.device_type == "cuda"
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)

        # --- Forcing ---
        if f[0] != "N/A":
            f = f.to(device, non_blocking=non_blocking)
            f_is_real = True
        else:
            B, Cy, *spatial = y.shape
            f = torch.zeros(B, Cy, *spatial, device=device)
            f_is_real = False

        # --- GPU-side normalization (replaces per-sample CPU loop in pydataset) ---
        x, y, _ = self._normalize_inputs(x=x, y=y)
        if f_is_real:
            _, _, f = self._normalize_inputs(f=f)

        optimizer.zero_grad(set_to_none=True)

        # --- Forward pass for each ensemble member + loss under AMP autocast ---
        with self._amp_ctx():
            prediction_list = []
            for m in range(members):
                pred_m = model(x, f)  # shape: (B, C, H, W) or (B, C, G)
                prediction_list.append(pred_m)
            # Stack along new ensemble dimension -> (B, M, C, H, W) or (B, M, C, G)
            prediction = torch.stack(prediction_list, dim=1)
            loss = loss_function(target=y, output=prediction)

        # --- Backpropagation (optimizer.step is handled by training_loop) ---
        if is_this_training:
            if self._scaler is not None:
                self._scaler.scale(loss).backward()
            else:
                loss.backward()

        return loss.detach()

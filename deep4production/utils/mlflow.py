import os
import mlflow
from deep4production.utils.general import get_func_from_string

# -------------------------------------------------------------------------
def mlflow_scalars_logs(tgt, prd, vars, mlflow_info, epoch):
    diagnostic_module = "deep4production.utils.diagnostics"
    for var in vars:
        kwargs = {"target": tgt[var], "prediction": prd[var]}
        diagnostics_to_run_scalars = []
        # Get diagnostics for this variable: "default"
        if "default" in mlflow_info:
            diagnostics_to_run_scalars.extend(mlflow_info["default"])
        # Get diagnostics for this variable: "variable-specific" (if available)
        if var in mlflow_info:
            diagnostics_to_run_scalars.extend(mlflow_info[var])
        for diagnostic in diagnostics_to_run_scalars:
            diagnostic_name = diagnostic
            if len(diagnostic) == 2:
                diagnostic_name = f"{diagnostic[0]}_{diagnostic[1]}"
                kwargs.update({"index": diagnostic[0]})
                diagnostic = diagnostic[1]
            value = get_func_from_string(diagnostic_module, diagnostic, kwargs = kwargs)
            mlflow.log_metric(f"{diagnostic_name}_{var}", float(value), step=int(epoch))
        print(f"🌐 (Mlflow) For VARIABLE: {var}\n"
            f"  --> The following SCALARS were LOGGED: {diagnostics_to_run_scalars}")

# -------------------------------------------------------------------------
def mlflow_figures_logs(tgt, prd, vars, mlflow_info, epoch):
    for var in vars:
        # collect all diagnostics for this variable
        diagnostics_to_run_figures = {}
        # add default diagnostics
        if "default" in mlflow_info:
            diagnostics_to_run_figures.update(mlflow_info["default"])
        # add variable-specific diagnostics
        if var in mlflow_info:
            diagnostics_to_run_figures.update(mlflow_info[var])
        logged = []  # track logged figures
        for diag_name, diag_cfg in diagnostics_to_run_figures.items():
            ## 1. Load diagnostic figure function   
            diag_func = get_func_from_string(diag_cfg["module"], diag_cfg["name"])
            # kwargs passed directly to the diagnostic plotting function
            fig_kwargs = diag_cfg.get("kwargs", {}).copy()
            ## 2. If an index function is defined, compute index first
            if "index" in diag_cfg:
                idx_cfg = diag_cfg["index"]
                index_func = get_func_from_string(idx_cfg["module"], idx_cfg["name"])
                index_kwargs = idx_cfg.get("kwargs", {})
                # compute index 
                index_target = index_func(tgt[var], **index_kwargs)
                index_prediction = index_func(prd[var], **index_kwargs)
                # the diagnostic function expects the index under key "index"
                fig_kwargs["data"] = [index_target, index_prediction]
            else:
                fig_kwargs.update({"data": [tgt[var], prd[var]]})
            ## 3. Compute the figure
            fig = diag_func(**fig_kwargs)
            ## 4. Log the figure in MLflow
            file_name = f"{diag_name}_epoch_{epoch:04d}.png"
            fig.savefig(file_name, bbox_inches="tight", dpi=300)
            mlflow.log_artifact(file_name, artifact_path=f"figures/{var}")
            os.remove(file_name)
            logged.append(diag_name)
        print(f"🌐 (Mlflow) For VARIABLE: {var}\n"
            f"  --> The following FIGURES were LOGGED: {logged}")


# -------------------------------------------------------------------------
# ## Log scalars (xai) ------------------------------------------------------------------------------
# Mlflow_scalars_xai = self.Mlflow_diagnostics.get("xai_scalars", None)
# if Mlflow_scalars_xai is not None:
#     x_mlflow = torch.cat([v[0] for v in valid_data], dim=0)
#     for i, var in enumerate(self.metadata_dict["vars_y"]):
#         kwargs_xai = {"x": x_mlflow.to(self.device), "model": self.model}
#         diagnostics_to_run_xai = {}
#         # Get diagnostics for this variable: "default"
#         if "default" in Mlflow_scalars_xai:
#             diagnostics_to_run_xai.update(Mlflow_scalars_xai["default"])
#         # Get diagnostics for this variable: "variable-specific" (if available)
#         if var in Mlflow_scalars_xai:
#             diagnostics_to_run_xai.update(Mlflow_scalars_xai[var])
#         for diag_name, diag_xai in diagnostics_to_run_xai.items():
#             kwargs_xai.update(**diag_xai.get("kwargs", None))
#             value = get_func_from_string(diag_xai["module"], diag_xai["name"], kwargs = kwargs_xai)
#             for c, var_x_name in enumerate(self.metadata_dict["vars_x"]):
#                 mlflow.log_metric(
#                     f"{diag_name}_{var}_{var_x_name}",     # label is here
#                     float(value[c]),
#                     step=int(epoch)
#             )
#             # mlflow.log_metric(f"{diag_name}_{var}", float(value), step=int(epoch))
#         print(f"🌐 (Mlflow) For VARIABLE: {var}\n"
#             f"  --> The following XAI-SCALARS were LOGGED: {diagnostics_to_run_xai}")
# Execute this file in the SAME notebook kernel after the verified interrupt:
# %run -i scripts/recover_notebook.py
import sys

_recovery_exception = getattr(sys, "last_value", None)
_recovery_traceback = getattr(sys, "last_traceback", None)
from policy_r1.recovery import save_interrupted_collection

saved_checkpoint = save_interrupted_collection(
    experiment, _recovery_exception, _recovery_traceback)
print("RECOVERY COMPLETE:", saved_checkpoint.resolve())

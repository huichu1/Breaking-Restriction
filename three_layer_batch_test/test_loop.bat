@echo off
for /L %%i in (1001,1,1100) do (
    echo --- Currently processing iteration: %%i ---
    python three_layer_all_layer_recovery.py 1 %%i
)
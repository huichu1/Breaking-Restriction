@echo off
for /L %%i in (1,1,100) do (
    echo --- Currently processing iteration: %%i ---
    python two_layer_all_layer_recovery.py 1 %%i
)
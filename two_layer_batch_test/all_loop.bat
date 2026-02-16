@echo off
for /L %%i in (1,1,100) do (
    echo --- Currently processing iteration: %%i ---
    python two_layer_find_duals.py 1 %%i
    python two_layer_find_duals.py 2 %%i
    python two_layer_find_duals.py 3 %%i
    python two_layer_cluster.py 0 %%i
    python two_layer_cluster.py 1 %%i
    python two_layer_all_layer_recovery.py 1 %%i
)
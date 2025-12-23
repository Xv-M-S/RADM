# run command

``` bash
export CUDA_VISIBLE_DEVICES=2
nohup python3 -u train_net.py \
     --num-gpus 1 \
     --config-file configs/radm.yaml \
     > ./log/dual_gram.log 2>&1 &

```

# warning 
if the env of radm is bad, then may be cause by install pytest.
you can try to install the typing_extensions 4.5.0 to sovle it.

``` bash
Installing collected packages: typing-extensions, pluggy, iniconfig, exceptiongroup, pytest
  Attempting uninstall: typing-extensions
    Found existing installation: typing_extensions 4.5.0
    Uninstalling typing_extensions-4.5.0:
      Successfully uninstalled typing_extensions-4.5.0
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tensorflow 2.13.1 requires typing-extensions<4.6.0,>=3.6.6, but you have typing-extensions 4.13.2 which is incompatible.
```
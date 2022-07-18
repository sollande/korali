. ~/.local/bin/ap.setenv
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/lib64



(cd hook && make) && make cleanrun bs="16 16 16" np=12
make cleanall


Please note that `LD_LIBRARY_PATH` needs to be set properly in GKE as per [Run GPUs in GKE](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus)


Following snippet is present in the deploymentyaml file.

```bash
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export PATH=$PATH:/usr/local/nvidia/bin:/usr/local/nvidia/lib64
/sbin/ldconfig
```
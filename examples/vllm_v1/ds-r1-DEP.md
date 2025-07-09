TODO


node 0
```bash
./launch/dsr1_dep.sh --num-nodes 2 --node-rank 0 --gpus-per-node 8 --master-addr <node 0 addr>
```

node 1
```bash
./launch/dsr1_dep.sh --num-nodes 2 --node-rank 1 --gpus-per-node 8 --master-addr <node 0 addr>
```
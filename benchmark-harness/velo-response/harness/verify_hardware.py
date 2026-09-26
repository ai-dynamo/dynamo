"""Verify the selected RDMA ports on each allocated node before freezing a run."""
import json
import os
from pathlib import Path
import socket
import subprocess

root=Path('/lustre/fsw/coreai_comparch_trtllm/jothomson/dynamo-velo-response-20260923')
result={'hostname':socket.gethostname(),'job':os.environ['SLURM_JOB_ID'],'ports':{}}
for node,device in enumerate(('mlx5_0','mlx5_4')):
    base=Path('/sys/class/infiniband')/device
    values={key:(base/'ports/1'/key).read_text().strip() for key in ('state','rate','link_layer')}
    values['numa_node']=int((base/'device/numa_node').read_text())
    assert values['numa_node']==node,values
    assert values['state']=='4: ACTIVE' and values['link_layer']=='InfiniBand',values
    result['ports'][device+':1']=values
result['ethernet']=subprocess.check_output(['ethtool','enP6p3s0f1np1'],text=True)
result['ethernet_mtu']=int(Path('/sys/class/net/enP6p3s0f1np1/mtu').read_text())
path=root/'manifests'/f'hardware-{result["job"]}-{result["hostname"]}.json'
path.write_text(json.dumps(result,indent=2)+'\n')
print(path)

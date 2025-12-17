from argparse import ArgumentParser
from pathlib import Path

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", type=str, required=True)
    parser.add_argument("output_dir", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    

    prefill_count = instance["prefill"]["num"]
    prefill_tp_size = instance["prefill"]["tensor_parallel_size"]
    decode_count = instance["decode"]["num"]
    decode_tp_size = instance["decode"]["tensor_parallel_size"]

    data = {
        "prefill_count": prefill_count,
        "decode_count": decode_count,
        "nodes_per_prefill": nodes_per_prefill,
        "nodes_per_decode": nodes_per_decode,
    }
    
    yaml.dump(data, open(Path(args.output_dir) / "instance_config.yaml", "w"))
    
        
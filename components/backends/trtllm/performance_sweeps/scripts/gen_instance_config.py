from argparse import ArgumentParser
from pathlib import Path
import yaml

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    

    prefill_count = config["prefill"]["num"]
    prefill_tp_size = config["prefill"]["config"]["tensor_parallel_size"]
    decode_count = config["decode"]["num"]
    decode_tp_size = config["decode"]["config"]["tensor_parallel_size"]

    data = {
        "prefill_count": prefill_count,
        "decode_count": decode_count,
        "prefill_tp_size": prefill_tp_size,
        "decode_tp_size": decode_tp_size
    }
    
    yaml.dump(data, open(Path(args.output_dir) / "instance_config.yaml", "w"))
    
        

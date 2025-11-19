import click
from dataclasses import dataclass

@dataclass
class ModelConfig:
    kind: str
    path: str
    served_name: str

@dataclass
class NodeConfig:
    num: int
    tp_size: int
    ep_size: int
    enable_attention_dp: bool
    batch_size: int
    max_num_tokens: int
    gpu_memory_fraction: float
    eplb_num_slots: int

@dataclass
class SweepConfig:
    image: str
    isl: int
    osl: int

@click.command()
@click.argument('model-kind', type=str, default='gpt-oss')
@click.argument('model-path', type=str, default='')
@click.argument('served-model-name', type=str, default='')
@click.argument('image', type=str, default='')
@click.option('--mtp', is_flag=True, default=False)
@click.option('--ctx_num', type=int, default=1)
@click.option('--ctx_tp_size', type=int, default=1)
@click.option('--ctx_ep_size', type=int, default=1)
@click.option('--ctx_batch_size', type=int, default=64)
@click.option('--ctx_max_num_tokens', type=int, default=20000)
@click.option('--ctx_enable_attention_dp', is_flag=True, default=False)
@click.option('--ctx_gpu_memory_fraction', type=float, default=0.9)
@click.option('--gen_num', type=int, default=1)
@click.option('--gen_tp_size', type=int, default=4)
@click.option('--gen_ep_size', type=int, default=1)
@click.option('--gen_enable_attention_dp', is_flag=True, default=False)
@click.option('--gen_batch_size', type=int, default=2048)
@click.option('--gen_max_num_tokens', type=int, default=1)
@click.option('--gen_gpu_memory_fraction', type=float, default=0.9)
@click.option('--gen_eplb_num_slots', type=int, default=0)
@click.option('--isl', type=int, default=8192)
@click.option('--osl', type=int, default=1024)
def main(**kwargs):
    model_cfg = ModelConfig(
        kind=kwargs['model_kind'],
        path=kwargs['model_path'],
        served_name=kwargs['served_model_name'],
        image=kwargs['image']
    )

    ctx_cfg = NodeConfig(
        num=kwargs['ctx_num'],
        tp_size=kwargs['ctx_tp_size'],
        ep_size=kwargs['ctx_ep_size'],
        batch_size=kwargs['ctx_batch_size'],
        max_num_tokens=kwargs['ctx_max_num_tokens'],
        enable_attention_dp=kwargs['ctx_enable_attention_dp'],
        gpu_memory_fraction=kwargs['ctx_gpu_memory_fraction'],
        eplb_num_slots=0
    )

    gen_cfg = NodeConfig(
        num=kwargs['gen_num'],
        tp_size=kwargs['gen_tp_size'],
        ep_size=kwargs['gen_ep_size'],
        enable_attention_dp=kwargs['gen_enable_attention_dp'],
        batch_size=kwargs['gen_batch_size'],
        max_num_tokens=kwargs['gen_max_num_tokens'],
        gpu_memory_fraction=kwargs['gen_gpu_memory_fraction'],
        eplb_num_slots=kwargs['gen_eplb_num_slots']
    )

    sweep_cfg = SweepConfig(
        image=kwargs['image'],
        isl=kwargs['isl'],
        osl=kwargs['osl']
    )

    # Example usage of the configs
    print(f"Running with config:\n{model_cfg}\n{ctx_cfg}\n{gen_cfg}\n{sweep_cfg}")

if __name__ == "__main__":
    main()

import os
from dataclasses import dataclass, field

from huggingface_hub import snapshot_download
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import FlexibleArgumentParser


@dataclass
class NvAsyncEngineArgs(AsyncEngineArgs):
    model_path: str = field(default="")


def parse_vllm_args() -> NvAsyncEngineArgs:
    parser = FlexibleArgumentParser()
    parser = AsyncEngineArgs.add_cli_args(parser)
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
    )
    args = parser.parse_args()
    if args.model_path == "":
        if os.environ.get("HF_TOKEN"):
            args.model_path = snapshot_download(args.model)
        else:
            raise ValueError(
                "Please set HF_TOKEN environment variable "
                "or pass --model-path to load the model"
            )
    return NvAsyncEngineArgs.from_cli_args(args)

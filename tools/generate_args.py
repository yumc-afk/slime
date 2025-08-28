import shlex
from typing import Sequence

from hydra import compose, initialize


def format_args(name: str, args: Sequence[str]) -> str:
    quoted = " ".join(shlex.quote(str(a)) for a in args)
    return f"{name}=(\n  {quoted}\n)"


def main():
    with initialize(config_path="../conf", job_name="generate_args", version_base=None):
        cfg = compose(config_name="run_qwen3_235B_A22B")

    for key, value in cfg.items():
        if isinstance(value, list):
            print(format_args(key.upper(), value))


if __name__ == "__main__":
    main()

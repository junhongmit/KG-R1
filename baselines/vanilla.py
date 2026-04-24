import argparse

from baselines.common import build_vanilla_prompt
from baselines.runner import add_common_args, run_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone KG-R1 vanilla baseline using an external OpenAI-compatible API.")
    add_common_args(parser, strategy_name="vanilla", default_n_rollouts=1)
    parser.set_defaults(max_tokens=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_baseline(
        args=args,
        strategy_name="vanilla",
        prompt_builder=build_vanilla_prompt,
        response_parser=lambda text: text.strip(),
        aggregate_fn=None,
    )


if __name__ == "__main__":
    main()

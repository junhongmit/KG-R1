import argparse

from baselines.common import build_cot_prompt, extract_final_answer
from baselines.runner import add_common_args, run_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone KG-R1 chain-of-thought baseline using an external OpenAI-compatible API.")
    add_common_args(parser, strategy_name="cot", default_n_rollouts=1)
    parser.set_defaults(max_tokens=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_baseline(
        args=args,
        strategy_name="cot",
        prompt_builder=build_cot_prompt,
        response_parser=extract_final_answer,
        aggregate_fn=None,
    )


if __name__ == "__main__":
    main()

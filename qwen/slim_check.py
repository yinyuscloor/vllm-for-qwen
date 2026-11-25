# SPDX-License-Identifier: Apache-2.0
"""Slim-round smoke check: the model must load and produce non-empty output."""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen3.8-27B",
                        help="HF id or local checkpoint path")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, max_model_len=args.max_model_len)
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    prompts = [
        "用一句话介绍杭州。",
        "1+1等于几？",
    ]
    outputs = llm.generate(prompts, params)

    ok = True
    for out in outputs:
        text = out.outputs[0].text.strip()
        print(f"PROMPT: {out.prompt}")
        print(f"OUTPUT: {text}")
        print("---")
        if not text:
            ok = False

    if not ok:
        print("SLIM_CHECK FAILED: empty generation output", file=sys.stderr)
        return 1
    print("SLIM_CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

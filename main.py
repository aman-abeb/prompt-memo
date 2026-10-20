"""PromptMemo runnable entrypoint and quickstart demonstration."""

import sys
from prompt_memo.models import Message
from prompt_memo.normalizer import build_canonical_representation, hash_prompt


def main() -> None:
    """Runs a quick local verification of canonical normalization and prompt hashing."""
    raw_messages = [
        Message(role="system", content="  You are a helpful   assistant.  "),
        Message(role="user", content="  Hello   world!  \n\n How are you doing?   "),
    ]
    equivalent_raw_messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="USER", content="Hello world!\nHow are you doing?"),
    ]

    model = "gpt-4o"
    hash1 = hash_prompt(raw_messages, model=model)
    hash2 = hash_prompt(equivalent_raw_messages, model=model)

    canonical_str = build_canonical_representation(raw_messages, model=model)

    print("PromptMemo CLI Demo")
    print("=" * 19)
    print("Canonical representation:")
    for line in canonical_str.splitlines():
        print(f"  {line}")
    print()
    print(f"Exact SHA-256 Hash: {hash1}")
    print(f"Invariant Match: {'SUCCESS' if hash1 == hash2 else 'FAILED'}")


if __name__ == "__main__":
    main()
    sys.exit(0)

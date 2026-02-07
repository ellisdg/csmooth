import argparse
import importlib
from .paths import DEFAULT_CONFIG_PATH

SCRIPTS = {
    "generate-designs": "generate_null_designs",
    "feat-null-first": "run_feat_null_firstlevel",
    "first-level-null": "run_first_level_null_glm",
    "analyze-null": "analyze_feat_null_results",
    "group-null": "run_group_null_glm",
    "pipeline": "run_fpr_pipeline",
}


def main():
    parser = argparse.ArgumentParser(prog="fpr", description="FPR tools dispatcher")
    parser.add_argument("command", choices=SCRIPTS.keys(), help="Which tool to run")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the subcommand")
    parsed = parser.parse_args()

    module = importlib.import_module(f".{{SCRIPTS[parsed.command]}}", package=__name__)
    if hasattr(module, "main"):
        module.main(parsed.args)
    else:
        raise SystemExit(f"Module {SCRIPTS[parsed.command]} missing main()")


if __name__ == "__main__":
    main()

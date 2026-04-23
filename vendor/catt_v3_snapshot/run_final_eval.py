import os
import runpy
import sys


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    eval_script = os.path.join(repo_root, "scripts", "eval", "eval_catt.py")

    if not os.path.exists(eval_script):
        raise FileNotFoundError(f"Eval script not found: {eval_script}")

    sys.path.insert(0, repo_root)
    sys.argv = [eval_script, *sys.argv[1:]]
    runpy.run_path(eval_script, run_name="__main__")


if __name__ == "__main__":
    main()

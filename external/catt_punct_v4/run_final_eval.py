import os
import runpy
import sys


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    eval_script = os.path.join(repo_root, "scripts", "eval", "eval_catt.py")
    catt_dir = os.path.abspath(os.path.dirname(__file__))
    default_checkpoint = os.path.join(catt_dir, "catt_eo_ft_balanced_v4_punct", "catt_eo_ft_best.pt")

    if not os.path.exists(eval_script):
        raise FileNotFoundError(f"Eval script not found: {eval_script}")

    argv = [eval_script, "--catt-dir", catt_dir]
    if "--checkpoint" not in sys.argv[1:]:
        if not os.path.exists(default_checkpoint):
            raise FileNotFoundError(
                "Default v4 punctuation-aware checkpoint not found:\n"
                f"  {default_checkpoint}\n"
                "Train v4 first, or pass --checkpoint explicitly."
            )
        argv.extend(["--checkpoint", default_checkpoint])
    argv.extend(sys.argv[1:])

    sys.path.insert(0, repo_root)
    sys.argv = argv
    runpy.run_path(eval_script, run_name="__main__")


if __name__ == "__main__":
    main()

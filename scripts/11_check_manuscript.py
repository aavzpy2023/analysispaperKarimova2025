"""
11_check_manuscript.py — Checks that the manuscript only uses artifacts that the pipeline produced.

Follows every \\input from the main .tex file and verifies that
  * each \\includegraphics file and each \\input file exists;
  * every project variable (a CamelCase macro such as \\MorganRTwoMean) used in the text is defined by one of the generated
    latex/*.tex files or by the preamble.
Exits with a non-zero status when something is missing, so a stale or absent figure/number fails the pipeline.
"""
import glob
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import ROOT, LOGS_DIR
from logger_utils import setup_logger, explain

STANDARD_CAMELCASE = {"LaTeX", "TeX", "BibTeX", "AA", "WriteBookmarks"}
DEFINE = re.compile(r"\\(?:newcommand|renewcommand|providecommand|DeclareRobustCommand)\*?\s*\{?\\([A-Za-z]+)|\\def\\([A-Za-z]+)")
USE = re.compile(r"(?<!\\)\\([A-Za-z]+)")  # a doubled backslash is a line break followed by text


def strip_comments(text):
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())


def resolve(path_str, base_dir):
    candidates = [path_str, path_str + ".tex"] if not os.path.splitext(path_str)[1] else [path_str]
    for c in candidates:
        full = os.path.join(base_dir, c)
        if os.path.isfile(full):
            return full
    return None


def main():
    setup_logger(os.path.join(LOGS_DIR, "11_check_manuscript.log"))
    explain("What this step does", """
        Final consistency check of the paper against the pipeline outputs. It lists every figure and every included file the
        manuscript refers to and confirms they exist, and it confirms that each result variable used in the text was generated.
        A failure here means the text refers to something the pipeline did not produce (or no longer produces).
    """)
    mains = [f for f in glob.glob(os.path.join(ROOT, "*.tex")) if r"\documentclass" in open(f, encoding="utf-8").read()]
    if len(mains) != 1:
        sys.exit(f"Expected exactly one main .tex file in {ROOT}, found {len(mains)}")
    main_tex = mains[0]

    texts, errors, graphics, queue, seen = {}, [], set(), [main_tex], set()
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        text = strip_comments(open(path, encoding="utf-8").read())
        texts[path] = text
        for rel in re.findall(r"\\input\{([^}]+)\}", text):
            target = resolve(rel, ROOT)
            if target is None:
                errors.append(f"missing \\input file '{rel}' (referenced in {os.path.relpath(path, ROOT)})")
            else:
                queue.append(target)
        for rel in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
            graphics.add(rel)

    for rel in sorted(graphics):
        found = any(os.path.isfile(os.path.join(ROOT, rel + ext)) for ext in ("", ".png", ".pdf", ".jpg", ".jpeg"))
        print(f"[{'OK' if found else 'MISSING'}] figure {rel}", flush=True)
        if not found:
            errors.append(f"missing figure '{rel}'")

    all_text = "\n".join(texts.values())
    for name in sorted(set(re.findall(r"\\(?:newcommand|providecommand)\*?\s*\{\\([A-Za-z]+[0-9][A-Za-z0-9]*)\}", all_text))):
        errors.append(f"variable name \\{name} contains a digit, which LaTeX cannot use in a command name")
    defined = {a or b for a, b in DEFINE.findall(all_text)}
    used = {m for m in USE.findall(all_text) if sum(ch.isupper() for ch in m) >= 2}
    undefined = sorted(used - defined - STANDARD_CAMELCASE)
    for name in undefined:
        errors.append(f"variable \\{name} is used but never defined")

    print(f"\n[CHECK] {len(texts)} files followed, {len(graphics)} figures, {len(used)} project variables used, "
          f"{len(undefined)} undefined", flush=True)
    if errors:
        print("\n[FAIL] The manuscript refers to artifacts that do not exist:", flush=True)
        for e in errors:
            print(f"  - {e}", flush=True)
        sys.exit(1)
    print("[OK] Every figure, included file, and variable used by the manuscript is produced by the pipeline.", flush=True)


if __name__ == "__main__":
    main()

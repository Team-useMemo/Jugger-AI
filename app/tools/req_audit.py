import os, ast, sys, re

# === 설정 ===
PROJECT_DIR = "app"  # 너의 소스 루트 폴더
IGNORE_DIR_NAMES = {
    ".git", ".venv", "venv", "env", ".env", "__pycache__", ".mypy_cache", ".ruff_cache",
    "node_modules", ".ipynb_checkpoints",
    # 너의 케이스:
    "jugger",        # app/jugger (가상환경 들어있음)
    "site-packages", # 안전망
}
REQUIREMENTS_FILE = "requirements.txt"

# 모듈명 -> PyPI 패키지명 매핑(일부 대표 케이스)
MODULE_TO_PACKAGE = {
    "dotenv": "python-dotenv",
    "dateutil": "python-dateutil",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",     # 쓰면
    "OpenSSL": "pyOpenSSL",
    "setuptools": "setuptools",
    "IPython": "ipython",
    "PIL": "pillow",
    "yaml": "PyYAML",
    "tiktoken": "tiktoken",
    "sentence_transformers": "sentence-transformers",
    "google": "google-generativeai",  # 너가 쓰는 경우에 한해 대표 패키지로 매핑
}

# 표준라이브러리 후보 (파이썬 제공)
STDLIB = set(getattr(sys, "stdlib_module_names", set()))
# 일부 과거/내장 네임 수동 제외 보정
STDLIB.update({
    "builtins", "exceptions", "gc", "itertools", "marshal", "posix", "pwd", "resource",
    "select", "sys", "time", "unicodedata", "xxlimited", "zipimport", "zlib",
    "asyncio", "typing", "dataclasses", "json", "re", "math", "statistics",
    "pathlib", "subprocess", "argparse", "collections", "functools", "logging",
    "unittest", "email", "http", "html", "urllib", "importlib", "inspect",
})

def iter_py_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        # 디렉토리 무시 적용
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIR_NAMES]
        for f in filenames:
            if f.endswith(".py"):
                yield os.path.join(dirpath, f)

def detect_imports():
    mods = set()
    for path in iter_py_files(PROJECT_DIR):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                tree = ast.parse(fh.read(), filename=path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for n in node.names:
                        mods.add(n.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        mods.add(node.module.split(".")[0])
        except Exception as e:
            print(f"⚠️ Skip {path}: {e}")
    return mods

def normalize_pkg_name(s: str) -> str:
    s = s.strip()
    s = re.split(r"[<>=!~\[]", s, 1)[0]  # extras/버전 잘라내기
    return s.lower().replace("_", "-")

def load_requirements(path: str):
    pkgs = set()
    if not os.path.exists(path):
        return pkgs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-r "):
                continue
            # 로컬 경로나 VCS는 스킵(필요시 처리)
            if "://" in line or line.startswith((".", "/")):
                continue
            name = normalize_pkg_name(line)
            pkgs.add(name)
    return pkgs

def map_module_to_package(mod: str) -> str:
    # 내부/숨김 모듈 스킵
    if mod.startswith("_"):
        return ""
    # 표준 라이브러리 제외
    if mod in STDLIB:
        return ""
    # 흔한 내장 네임 추가 필터
    if mod in {"app", "tests", "test", "models", "utils", "config"}:
        return ""
    # 특징적 매핑
    if mod in MODULE_TO_PACKAGE:
        return normalize_pkg_name(MODULE_TO_PACKAGE[mod])
    # heuristic: 그냥 모듈명 그대로 패키지명 후보
    return normalize_pkg_name(mod)

def main():
    detected_modules = detect_imports()
    # 모듈 → 패키지
    detected_pkgs = set()
    for m in detected_modules:
        p = map_module_to_package(m)
        if p:
            detected_pkgs.add(p)

    req_pkgs = load_requirements(REQUIREMENTS_FILE)

    add_these = sorted(detected_pkgs - req_pkgs)
    maybe_unused = sorted(req_pkgs - detected_pkgs)

    print("\n📦 Detected top-level modules:", sorted(detected_modules))
    print("\n✅ Packages inferred from imports:", sorted(detected_pkgs))
    print("\n📝 Packages in requirements.txt:", sorted(req_pkgs))

    if add_these:
        print("\n➕ Add to requirements.txt (missing):")
        for p in add_these:
            print("  -", p)
    else:
        print("\n➕ Missing: none")

    if maybe_unused:
        print("\n➖ Possibly unused (present in requirements.txt but not detected in imports):")
        for p in maybe_unused:
            print("  -", p)
    else:
        print("\n➖ Unused: none")

if __name__ == "__main__":
    main()

import ast
import shutil
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = PROJECT_ROOT / "dist"
HANDOFF_DIR = DIST_DIR / "cloth_handoff"
ZIP_PATH = DIST_DIR / "cloth_handoff.zip"

ROOT_FILES = [
    "README.md",
    "requirements.txt",
    ".env.example",
    ".gitignore",
    "run_local.ps1",
    "run_lan.ps1",
    "setup_models.ps1",
    "sam2_refiner.py",
]

SCRIPT_FILES = [
    "scripts/download_models.py",
    "scripts/package_handoff.py",
]

DOC_FILES = [
    "docs/MODULE_GUIDE.md",
    "docs/TEACHER_QUICK_START.md",
]

NN_FILES = [
    "nn/config.py",
    "nn/diffusion_model.py",
    "nn/garment_parser.py",
    "nn/gradioFinal.py",
    "nn/model.py",
]

PROHIBITED_SUBSTRINGS = [
    "hub/",
    "nn/result/",
    "result/",
    "__pycache__/",
    ".vscode/",
    "result_temp.jpg",
    "test_",
    "sam2_test.py",
    "nn/fashn_parser_test.py",
    "nn/test",
    "nn/sam2/checkpoints/",
    "nn/Self-Correction-Human-Parsing/checkpoints/",
]

PROHIBITED_BASENAME_GLOBS = [
    "debug_*.png",
]

PROHIBITED_SUFFIXES = [
    ".pt",
    ".pth",
    ".bin",
    ".ckpt",
    ".safetensors",
]

PROHIBITED_EXACT_NAMES = [
    ".env",
]


def iter_local_imported_files(source_file: Path) -> list[str]:
    source_text = source_file.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(source_text, filename=str(source_file))
    package_root = source_file.parent
    found: set[str] = set()

    def try_add(module_name: str, base_dir: Path) -> None:
        candidate = (base_dir / Path(*module_name.split("."))).with_suffix(".py")
        if candidate.exists():
            found.add(candidate.relative_to(PROJECT_ROOT).as_posix())

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                try_add(alias.name, PROJECT_ROOT)
                try_add(alias.name, PROJECT_ROOT / "nn")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                if node.level > 0:
                    resolved_base = package_root
                    for _ in range(node.level - 1):
                        resolved_base = resolved_base.parent
                    try_add(node.module, resolved_base)
                else:
                    try_add(node.module, PROJECT_ROOT)
                    try_add(node.module, PROJECT_ROOT / "nn")
            else:
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    if node.level > 0:
                        resolved_base = package_root
                        for _ in range(node.level):
                            resolved_base = resolved_base.parent
                        try_add(alias.name, resolved_base)
    return sorted(found)


def build_manifest_list() -> list[str]:
    manifest = []
    manifest.extend(ROOT_FILES)
    manifest.extend(SCRIPT_FILES)
    manifest.extend(DOC_FILES)
    manifest.extend(NN_FILES)
    manifest.extend(iter_local_imported_files(PROJECT_ROOT / "nn" / "gradioFinal.py"))
    deduped_manifest = []
    seen = set()
    for item in manifest:
        if item not in seen:
            seen.add(item)
            deduped_manifest.append(item)
    return deduped_manifest


def reset_dist() -> None:
    if HANDOFF_DIR.exists():
        shutil.rmtree(HANDOFF_DIR)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)


def copy_manifest_files(manifest: list[str]) -> None:
    for rel_path in manifest:
        src = PROJECT_ROOT / rel_path
        if not src.exists():
            raise FileNotFoundError(f"Required file not found: {rel_path}")
        dst = HANDOFF_DIR / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_manifest(manifest: list[str]) -> None:
    manifest_path = HANDOFF_DIR / "MANIFEST.txt"
    manifest_text = "\n".join(manifest) + "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8")


def assert_clean_tree(base_dir: Path) -> None:
    for path in base_dir.rglob("*"):
        rel = path.relative_to(base_dir).as_posix()
        lower_rel = rel.lower()
        basename = Path(rel).name
        if basename.lower() in PROHIBITED_EXACT_NAMES:
            raise RuntimeError(f"Prohibited content found in handoff tree: {rel}")
        for bad in PROHIBITED_SUBSTRINGS:
            if bad.lower() in lower_rel:
                raise RuntimeError(f"Prohibited content found in handoff tree: {rel}")
        for bad_glob in PROHIBITED_BASENAME_GLOBS:
            if Path(basename).match(bad_glob):
                raise RuntimeError(f"Prohibited content found in handoff tree: {rel}")
        if path.is_file() and path.suffix.lower() in PROHIBITED_SUFFIXES:
            raise RuntimeError(f"Prohibited file type found in handoff tree: {rel}")


def build_zip() -> None:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    with ZipFile(ZIP_PATH, "w", compression=ZIP_DEFLATED) as zip_file:
        for file_path in HANDOFF_DIR.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(HANDOFF_DIR).as_posix()
                zip_file.write(file_path, arcname)


def assert_clean_zip() -> None:
    with ZipFile(ZIP_PATH, "r") as zip_file:
        for name in zip_file.namelist():
            lower_name = name.lower()
            basename = Path(name).name
            if basename.lower() in PROHIBITED_EXACT_NAMES:
                raise RuntimeError(f"Prohibited content found in zip: {name}")
            for bad in PROHIBITED_SUBSTRINGS:
                if bad.lower() in lower_name:
                    raise RuntimeError(f"Prohibited content found in zip: {name}")
            for bad_glob in PROHIBITED_BASENAME_GLOBS:
                if Path(basename).match(bad_glob):
                    raise RuntimeError(f"Prohibited content found in zip: {name}")
            suffix = Path(name).suffix.lower()
            if suffix in PROHIBITED_SUFFIXES:
                raise RuntimeError(f"Prohibited file type found in zip: {name}")


def main() -> int:
    manifest = build_manifest_list()
    reset_dist()
    copy_manifest_files(manifest)
    write_manifest(manifest)
    assert_clean_tree(HANDOFF_DIR)
    build_zip()
    assert_clean_zip()

    included_count = len(
        [path for path in HANDOFF_DIR.rglob("*") if path.is_file()]
    )
    print(f"[Package] handoff folder path: {HANDOFF_DIR}")
    print(f"[Package] zip path: {ZIP_PATH}")
    print(f"[Package] included file count: {included_count}")
    print("[Package] handoff package created successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())

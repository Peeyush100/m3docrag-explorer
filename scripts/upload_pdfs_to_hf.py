"""One-time script: publish the M3DocVQA source PDFs to a Hugging Face
Datasets repo so the deployed Streamlit app can lazily fetch individual
pages without bundling ~1.6GB of PDFs into the git repo.

Usage:
    hf auth login        # paste a WRITE token
    python scripts/upload_pdfs_to_hf.py \
        --repo-id <your-hf-username>/m3docvqa-pdfs \
        --source-dir <path-to>/m3docvqa/data/pdfs_dev

`upload_large_folder` (the resumable, parallel uploader you want for
thousands of files) has no `path_in_repo` option, so to land the files under
`pdfs/` in the repo we first stage them in a temp dir under a `pdfs/`
subfolder. Staging uses hardlinks where possible, so it costs no extra disk.

Re-running is safe: files already on the Hub with identical content are
skipped.
"""

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi


def stage_pdfs(source_dir: Path, staging_root: Path) -> Path:
    """Mirror source_dir/*.pdf into staging_root/pdfs/ via hardlinks."""
    pdf_dir = staging_root / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(source_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No *.pdf files found in {source_dir}")

    linked = 0
    for src in pdfs:
        dst = pdf_dir / src.name
        if dst.exists():
            continue
        # resolve() so a symlinked corpus still hardlinks to the real file
        real = src.resolve()
        try:
            os.link(real, dst)
        except OSError:
            # different filesystem (or hardlink limit) - fall back to a copy
            shutil.copy2(real, dst)
        linked += 1

    print(f"Staged {len(pdfs)} PDFs ({linked} newly linked) at {pdf_dir}")
    return pdf_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="e.g. your-username/m3docvqa-pdfs")
    parser.add_argument(
        "--source-dir", required=True, help="Directory of *.pdf files (e.g. m3docvqa/data/pdfs_dev)"
    )
    parser.add_argument(
        "--staging-dir",
        default=None,
        help="Where to stage the pdfs/ tree (default: <source-dir>/../_hf_upload_staging)",
    )
    parser.add_argument("--private", action="store_true", help="Create the dataset repo as private")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    source_dir = Path(args.source_dir).expanduser()
    staging_root = (
        Path(args.staging_dir).expanduser()
        if args.staging_dir
        else source_dir.parent / "_hf_upload_staging"
    )

    stage_pdfs(source_dir, staging_root)

    api = HfApi()
    print(f"Creating (or reusing) dataset repo: {args.repo_id}")
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)

    print(f"Uploading -> {args.repo_id} (files will land under pdfs/) ...")
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(staging_root),
        allow_patterns=["pdfs/*.pdf"],
        num_workers=args.num_workers,
    )
    print("\nDone. Point the app at this dataset with:")
    print(f"    export HF_PDF_DATASET_REPO={args.repo_id}")


if __name__ == "__main__":
    main()

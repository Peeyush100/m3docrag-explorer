"""One-time script: publish the M3DocVQA source PDFs to a public Hugging
Face Datasets repo so the deployed Streamlit app can lazily fetch individual
pages without bundling ~1.6GB of PDFs into the git repo.

Usage:
    huggingface-cli login        # or: hf auth login  (paste a WRITE token)
    python scripts/upload_pdfs_to_hf.py \
        --repo-id <your-hf-username>/m3docvqa-pdfs \
        --source-dir <path-to>/m3docvqa/data/pdfs_dev

This only needs to be run once (and again if you add more PDFs later -
upload_large_folder skips files that already exist on the Hub with the same
content).
"""

import argparse

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="e.g. your-username/m3docvqa-pdfs")
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Local directory of *.pdf files (e.g. m3docvqa/data/pdfs_dev)",
    )
    parser.add_argument("--private", action="store_true", help="Create the dataset repo as private")
    args = parser.parse_args()

    api = HfApi()
    print(f"Creating (or reusing) dataset repo: {args.repo_id}")
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)

    print(f"Uploading PDFs from {args.source_dir} -> {args.repo_id}:pdfs/ ...")
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=args.source_dir,
        path_in_repo="pdfs",
        allow_patterns=["*.pdf"],
    )
    print("Done. Set HF_PDF_DATASET_REPO to this repo id when running the app:")
    print(f"    export HF_PDF_DATASET_REPO={args.repo_id}")


if __name__ == "__main__":
    main()

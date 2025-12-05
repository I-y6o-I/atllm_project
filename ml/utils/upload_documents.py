import os
import sys
import argparse
import requests
from pathlib import Path
from typing import Optional
import time
import json

DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost/api/v1")
DEFAULT_TIMEOUT = 60  # seconds
DEFAULT_MAPPING_FILE = "/home/kirill/Files/IU_Cources/atllm_project/ml/data/id_mapping.json"

# python ml/utils/upload_documents.py -d ml/data/qasper_pdfs -m ml/data/id_mapping.json -t eyJhbGciOiJIUzI1NiJ9.eyJyb2xlIjoiUk9MRV9VU0VSIiwidG9rZW5JZCI6IjViYThmYmE2LWMzY2ItNDdjMS04MjM5LTBiOTlmMWNkNmIwZCIsImlkIjoxLCJlbWFpbCI6ImRlbW91c2VyIiwic3ViIjoiZGVtb3VzZXIiLCJpYXQiOjE3NjQ5NjU5MTgsImV4cCI6MTc2NTEwOTkxOH0.USDtsXhmCj7_gkrZ3k45IGOsfBQBDeROa7FLls5akNI

def load_qasper_dataset() -> dict:
    """
    Load allenai/qasper dataset and create a lookup by arxiv ID.
    Returns dict: {arxiv_id: {"title": ..., "abstract": ...}, ...}
    """
    print("Loading allenai/qasper dataset from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/qasper", split="train")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Falling back to empty lookup - will use filename as title")
        return {}
    
    lookup = {}
    for paper in ds:
        arxiv_id = paper.get("id")
        if arxiv_id:
            lookup[arxiv_id] = {
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", "")
            }
    
    print(f"Loaded {len(lookup)} papers from dataset")
    return lookup


def load_id_mapping(mapping_file: Path) -> dict:
    """
    Load id_mapping.json: {local_id: arxiv_id, ...}
    e.g., {"001": "1234.56789", "002": "2345.67890", ...}
    """
    if not mapping_file.exists():
        print(f"Warning: Mapping file not found: {mapping_file}")
        return {}
    
    with open(mapping_file, "r") as f:
        mapping = json.load(f)
    
    print(f"Loaded {len(mapping)} mappings from {mapping_file}")
    return mapping


def get_auth_token() -> Optional[str]:
    """
    Get auth token from environment or return None.
    Set AUTH_TOKEN env var or modify this function to authenticate.
    """
    return os.environ.get("AUTH_TOKEN")


def upload_article(
    api_url: str,
    pdf_path: Path,
    title: str,
    short_desc: str,
    token: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT
) -> dict:
    """
    Upload a single article PDF.
    
    Args:
        api_url: Base API URL (e.g., http://localhost/api/v1)
        pdf_path: Path to the PDF file
        title: Article title
        short_desc: Short description (abstract)
        token: Optional auth token
        timeout: Request timeout in seconds
    
    Returns:
        API response as dict
    """
    url = f"{api_url.rstrip('/')}/articles"
    
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    with open(pdf_path, "rb") as f:
        files = {
            "pdf_file": (pdf_path.name, f, "application/pdf")
        }
        data = {
            "title": title,
            "short_desc": short_desc[:999]
        }
        
        response = requests.post(
            url,
            files=files,
            data=data,
            headers=headers,
            timeout=timeout
        )
    
    response.raise_for_status()
    return response.json()


def trigger_ml_indexing(
    api_url: str,
    article_id: str,
    token: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT
) -> Optional[dict]:
    """
    Trigger ML service to index the uploaded article.
    Non-blocking, errors are logged but don't stop the process.
    """
    url = f"{api_url.rstrip('/')}/ml/index_assignment"
    
    # headers = {"Content-Type": "application/json"}
    # if token:
    #     headers["Authorization"] = f"Bearer {token}"
    
    try:
        response = requests.post(
            url,
            data={"assignment_id": str(article_id)},
            # headers=headers,
            timeout=timeout
        )
        if response.status_code != 204:
            print(f"  [WARN] ML indexing returned status {response.status_code} for article {article_id}")
            return None
    except Exception as e:
        print(f"  [WARN] ML indexing failed for article {article_id}: {e}")
        return None


def get_sorted_pdfs(directory: Path) -> list[Path]:
    """Get all PDF files in directory, sorted by filename."""
    pdfs = list(directory.glob("*.pdf"))
    pdfs.sort(key=lambda p: p.stem)  # Sort by filename without extension
    return pdfs


def derive_metadata_from_dataset(
    pdf_path: Path,
    id_mapping: dict,
    dataset_lookup: dict
) -> tuple[str, str]:
    """
    Derive title and abstract from HuggingFace dataset using id_mapping.
    
    Args:
        pdf_path: Path to PDF file (e.g., /path/to/001.pdf)
        id_mapping: {local_id: arxiv_id} mapping
        dataset_lookup: {arxiv_id: {"title": ..., "abstract": ...}} from HF dataset
    
    Returns:
        (title, short_desc) tuple
    """
    local_id = pdf_path.stem  # e.g., "001"
    arxiv_id = id_mapping.get(local_id)
    
    if not arxiv_id:
        # Fallback: use filename as title
        title = local_id.replace("_", " ").replace("-", " ")
        short_desc = f"Paper ID: {local_id}"
        return title, short_desc
    
    paper_info = dataset_lookup.get(arxiv_id, {})
    
    title = paper_info.get("title", "").strip()
    abstract = paper_info.get("abstract", "").strip()
    
    # Fallback if title is empty
    if not title:
        title = f"Paper {arxiv_id}"
    
    # Fallback if abstract is empty
    if not abstract:
        abstract = f"ArXiv ID: {arxiv_id}"
    
    # Truncate abstract if too long (adjust limit as needed)
    max_abstract_len = 2000
    if len(abstract) > max_abstract_len:
        abstract = abstract[:max_abstract_len - 3] + "..."
    
    return title, abstract


def batch_upload(
    pdf_dir: Path,
    api_url: str,
    id_mapping: dict,
    dataset_lookup: dict,
    token: Optional[str] = None,
    index_ml: bool = True,
    delay: float = 0.5,
    dry_run: bool = False
) -> dict:
    """
    Upload all PDFs from a directory.
    
    Args:
        pdf_dir: Directory containing PDF files
        api_url: Base API URL
        id_mapping: {local_id: arxiv_id} mapping
        dataset_lookup: {arxiv_id: {"title": ..., "abstract": ...}} from HF dataset
        token: Optional auth token
        index_ml: Whether to trigger ML indexing after upload
        delay: Delay between uploads in seconds
        dry_run: If True, only print what would be done
    
    Returns:
        Summary dict with success/failure counts
    """
    pdfs = get_sorted_pdfs(pdf_dir)
    
    if not pdfs:
        print(f"No PDF files found in {pdf_dir}")
        return {"total": 0, "success": 0, "failed": 0}
    
    print(f"\nFound {len(pdfs)} PDF files to upload")
    print(f"API URL: {api_url}")
    print(f"ML Indexing: {'enabled' if index_ml else 'disabled'}")
    print("-" * 60)
    
    results = {
        "total": len(pdfs),
        "success": 0,
        "failed": 0,
        "uploaded": [],
        "errors": []
    }
    
    for i, pdf_path in enumerate(pdfs, 1):
        title, short_desc = derive_metadata_from_dataset(
            pdf_path, id_mapping, dataset_lookup
        )
        
        print(f"\n[{i}/{len(pdfs)}] Uploading: {pdf_path.name}")
        print(f"  Title: {title[:80]}{'...' if len(title) > 80 else ''}")
        print(f"  Abstract: {short_desc[:100]}{'...' if len(short_desc) > 100 else ''}")
        
        if dry_run:
            print("  [DRY RUN] Skipping actual upload")
            results["success"] += 1
            continue
        
        try:
            # Upload article
            response = upload_article(
                api_url=api_url,
                pdf_path=pdf_path,
                title=title,
                short_desc=short_desc,
                token=token
            )
            
            article_id = response.get("id")
            print(f"  [OK] Created article ID: {article_id}")
            
            # Trigger ML indexing
            if index_ml and article_id:
                print(f"  Triggering ML indexing...")
                trigger_ml_indexing(api_url, article_id, token)
            
            results["success"] += 1
            results["uploaded"].append({
                "file": pdf_path.name,
                "id": article_id,
                "title": title,
                "arxiv_id": id_mapping.get(pdf_path.stem)
            })
            
        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            try:
                error_msg = e.response.json()
            except:
                pass
            print(f"  [ERROR] HTTP Error: {error_msg}")
            results["failed"] += 1
            results["errors"].append({
                "file": pdf_path.name,
                "error": str(error_msg)
            })
            
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            results["failed"] += 1
            results["errors"].append({
                "file": pdf_path.name,
                "error": str(e)
            })
        
        # Delay between uploads to avoid overwhelming the server
        if delay > 0 and i < len(pdfs):
            time.sleep(delay)

        # break
    
    print("\n" + "-" * 60)
    print(f"Upload complete: {results['success']}/{results['total']} successful, {results['failed']} failed")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch upload PDF articles with metadata from HuggingFace dataset"
    )
    parser.add_argument(
        "--dir", "-d",
        type=Path,
        required=True,
        help="Directory containing PDF files to upload"
    )
    parser.add_argument(
        "--mapping", "-m",
        type=Path,
        default=Path(DEFAULT_MAPPING_FILE),
        help=f"Path to id_mapping.json (default: {DEFAULT_MAPPING_FILE})"
    )
    parser.add_argument(
        "--api-url", "-u",
        type=str,
        default=DEFAULT_API_URL,
        help=f"Base API URL (default: {DEFAULT_API_URL})"
    )
    parser.add_argument(
        "--token", "-t",
        type=str,
        default=None,
        help="Auth token (or set AUTH_TOKEN env var)"
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Skip ML indexing after upload"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between uploads in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done, don't actually upload"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output JSON file for results"
    )
    
    args = parser.parse_args()
    
    if not args.dir.is_dir():
        print(f"Error: {args.dir} is not a directory")
        sys.exit(1)
    
    # Load dataset and mapping
    id_mapping = load_id_mapping(args.mapping)
    dataset_lookup = load_qasper_dataset()
    
    token = args.token or get_auth_token()
    
    results = batch_upload(
        pdf_dir=args.dir,
        api_url=args.api_url,
        id_mapping=id_mapping,
        dataset_lookup=dataset_lookup,
        token=token,
        index_ml=not args.no_index,
        delay=args.delay,
        dry_run=args.dry_run
    )
    
    # Save results to JSON if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    
    # Exit with error code if any uploads failed
    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
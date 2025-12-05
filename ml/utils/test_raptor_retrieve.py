import os
import sys
import argparse
import requests
import json

DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost/api/v1")
DEFAULT_TIMEOUT = 30

def build_endpoint(base: str) -> str:
    b = base.rstrip("/")
    if b.endswith("/ml") or b.endswith("/api/v1/ml"):
        return f"{b}/raptor_search"
    return f"{b}/ml/raptor_search"

def call_raptor_search(api_base: str, query: str, paper_id: str = None,
                       mode: str = "collapsed", level: int = None,
                       limit: int = 10, score_threshold: float = 0.0,
                    timeout: int = DEFAULT_TIMEOUT):
    url = build_endpoint(api_base)
    params = {"query": query, "mode": mode, "limit": limit, "score_threshold": score_threshold}
    if paper_id:
        params["paper_id"] = paper_id
    if mode == "level" and level is not None:
        params["level"] = level

    # headers = {}

    resp = requests.get(url, params=params, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        print(f"HTTP {resp.status_code} error: {resp.text}", file=sys.stderr)
        resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        return resp.text

def main():
    p = argparse.ArgumentParser(description="Test /raptor_search endpoint")
    p.add_argument("--api-url", "-u", default=DEFAULT_API_URL, help="Base API URL (e.g. http://localhost/api/v1)")
    p.add_argument("--query", "-q", required=True, help="Query string")
    p.add_argument("--paper-id", "-p", default=None, help="Optional paper_id to filter")
    p.add_argument("--mode", "-m", choices=["collapsed","level"], default="collapsed")
    p.add_argument("--level", type=int, default=None, help="Level (required if mode=level)")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--score-threshold", type=float, default=0.0)
    # p.add_argument("--token", "-t", default=os.environ.get("AUTH_TOKEN"))
    p.add_argument("--raw", action="store_true", help="Print raw response")
    args = p.parse_args()

    if args.mode == "level" and args.level is None:
        p.error("mode=level requires --level")

    try:
        res = call_raptor_search(
            api_base=args.api_url,
            query=args.query,
            paper_id=args.paper_id,
            mode=args.mode,
            level=args.level,
            limit=args.limit,
            score_threshold=args.score_threshold,
            # token=args.token
        )
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        sys.exit(2)

    if args.raw:
        if isinstance(res, (dict, list)):
            print(json.dumps(res, indent=2))
        else:
            print(res)
    else:
        # Pretty summary
        if isinstance(res, list):
            print(f"Returned {len(res)} items")
            for i, item in enumerate(res, 1):
                nid = item.get("node_id") if isinstance(item, dict) else None
                score = item.get("score") if isinstance(item, dict) else None
                text = item.get("text", "") if isinstance(item, dict) else str(item)
                meta = item.get("metadata", {}) if isinstance(item, dict) else {}
                print(f"\n[{i}] node_id={nid} score={score}")
                print(f"metadata: {json.dumps(meta)}")
                print(f"text (snippet): {text[:400]}")
        else:
            print(res)

            
# python ml/utils/test_raptor_retrieve.py -q "Summarize method of Recurrent Networks" --mode collapsed

if __name__ == "__main__":
    main()
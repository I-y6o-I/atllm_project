#!/usr/bin/env python3
"""
Test script for RAPTOR build_tree endpoint.
"""

import requests
import json

BASE_URL = "http://localhost/api/v1/ml"



def test_build_tree(max_levels: int = 3, min_nodes_per_level: int = 3):
    print("\n=== Testing Build Tree ===")
    print(f"Parameters: max_levels={max_levels}, min_nodes_per_level={min_nodes_per_level}")
    
    response = requests.post(
        f"{BASE_URL}/build_tree",
        params={
            "max_levels": max_levels,
            "min_nodes_per_level": min_nodes_per_level
        }
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n--- Build Result ---")
        print(json.dumps(result, indent=2))
        
        if "final_stats" in result:
            stats = result["final_stats"]
            print(f"\n--- Tree Statistics ---")
            print(f"Total nodes: {stats.get('total_nodes', 'N/A')}")
            print(f"Max level: {stats.get('max_level', 'N/A')}")
            
            for level, info in stats.get("levels", {}).items():
                print(f"  Level {level}: {info.get('count', 0)} nodes ({info.get('node_type', '')})")
        
        return True
    else:
        print(f"Error: {response.text}")
        return False


def test_raptor_search(query: str = "machine learning", limit: int = 5):
    print(f"\n=== Testing RAPTOR Search ===")
    print(f"Query: '{query}'")
    
    response = requests.get(
        f"{BASE_URL}/raptor_search",
        params={
            "query": query,
            "mode": "collapsed",
            "level_boost": 100,
            # "level": 1,
            "limit": limit,
            "score_threshold": 0.0
        }
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        results = response.json()
        print(f"Found {len(results)} results\n")
        


        
        for i, hit in enumerate(results, 1):
            metadata = hit.get("metadata", {})
            # if metadata.get("node_type", "N/A") == "chunk":
            #     continue
            print(f"--- Result {i} ---")
            print(f"Score: {hit.get('score', 'N/A'):.4f}")
            print(f"Level: {metadata.get('level', 'N/A')}")
            print(f"Type: {metadata.get('node_type', 'N/A')}")
            # print("============================")
            # with open(f"/home/kirill/Files/IU_Cources/atllm_project/ml/data/summarizer_responses/response_{i}.txt", "w") as f:
            #     f.write(hit.get("text", ""))
            print(f"Text: {hit.get('text', '')[:300]}")
            print()
        
        return True
    else:
        print(f"Error: {response.text}")
        return False


def main():
    print("RAPTOR Build Tree Test Script")
    print("=" * 40)
    
    
    # test_build_tree(max_levels=3, min_nodes_per_level=3)
    
    test_raptor_search("Summarize papers about hate speech", limit=15)
    
    print("\n=== All tests completed ===")


if __name__ == "__main__":
    main()

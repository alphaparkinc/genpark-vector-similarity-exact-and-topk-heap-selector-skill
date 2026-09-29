import sys
import json
from client import TopKHeapSelector

def handle_request(req):
    method = req.get("method")
    req_id = req.get("id")
    
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "genpark-vector-similarity-exact-and-topk-heap-selector-skill", "version": "1.0.0"}
            }
        }
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "select_topk",
                        "description": "Selects top-K highest similarity candidates using a bounded min-heap",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query_vector": {"type": "array", "items": {"type": "number"}},
                                "candidates": {"type": "object"},
                                "k": {"type": "integer", "default": 3},
                                "metric": {"type": "string", "enum": ["cosine", "euclidean"], "default": "cosine"}
                            },
                            "required": ["query_vector", "candidates"]
                        }
                    }
                ]
            }
        }
    elif method == "tools/call":
        params = req.get("params", {})
        name = params.get("name")
        args = params.get("arguments", {})
        
        if name == "select_topk":
            res = TopKHeapSelector.select_topk(
                args.get("query_vector", []),
                args.get("candidates", {}),
                args.get("k", 3),
                args.get("metric", "cosine")
            )
            return {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}
            
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}

def run():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_request(req)
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "error": {"code": -32000, "message": str(e)}}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    run()

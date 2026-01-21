import json
import uuid
import time
import os
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime

class MemoryManager:
    """
    Manages persistent memory logs for the recommendation system.
    Stores traces, predictions, and training metadata.
    Current implementation uses a local JSONL file.
    """
    
    def __init__(self, storage_path: str = "data/memory_log.jsonl"):
        self.storage_path = storage_path
        self._write_lock = threading.Lock()  # Prevent concurrent write corruption
        self._ensure_storage_exists()
        
    def _ensure_storage_exists(self):
        """Ensure the storage directory and file exist."""
        directory = os.path.dirname(self.storage_path)
        if directory:  # Only create directory if path is not empty
            os.makedirs(directory, exist_ok=True)
        
        if not os.path.exists(self.storage_path):
            # Touch-style create without leaving open handle
            open(self.storage_path, 'a').close()

    def start_trace(self, request_data: Dict[str, Any]) -> str:
        """
        Start a new trace for an incoming request.
        
        Args:
            request_data: Dictionary containing request details (user_id, inputs, etc.)
            
        Returns:
            trace_id: Unique identifier for this trace
        """
        trace_id = str(uuid.uuid4())
        
        entry = {
            "trace_id": trace_id,
            "timestamp": datetime.utcnow().isoformat(),
            "type": "request",
            "data": request_data
        }
        
        self._append_log(entry)
        return trace_id

    def log_prediction(self, trace_id: str, prediction_data: Dict[str, Any]):
        """Log model prediction results associated with a trace."""
        entry = {
            "trace_id": trace_id,
            "timestamp": datetime.utcnow().isoformat(),
            "type": "prediction",
            "data": prediction_data
        }
        self._append_log(entry)

    def log_training(self, run_id: str, metrics: Dict[str, Any], params: Dict[str, Any]):
        """Log a model training run."""
        entry = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "type": "training",
            "metrics": metrics,
            "params": params
        }
        self._append_log(entry)

    def log_external_api(self, trace_id: Optional[str], api_name: str, status: str, latency: float):
        """Log an external API interaction."""
        entry = {
            "trace_id": trace_id,
            "timestamp": datetime.utcnow().isoformat(),
            "type": "external_api",
            "api": api_name,
            "status": status,
            "latency_ms": round(latency * 1000, 2)
        }
        self._append_log(entry)

    def get_trace(self, trace_id: str) -> Dict[str, Any]:
        """
        Retrieve full context for a specific trace ID.
        Aggregates request, prediction, and external calls.
        """
        trace_context = {
            "trace_id": trace_id,
            "request": None,
            "prediction": None,
            "external_calls": []
        }
        
        try:
            with open(self.storage_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("trace_id") == trace_id:
                            entry_type = entry.get("type")
                            if entry_type == "request":
                                trace_context["request"] = entry.get("data")
                                trace_context["timestamp"] = entry.get("timestamp")
                            elif entry_type == "prediction":
                                trace_context["prediction"] = entry.get("data")
                            elif entry_type == "external_api":
                                trace_context["external_calls"].append(entry)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            return {}
            
        return trace_context

    def _append_log(self, entry: Dict[str, Any]):
        """Append a log entry to the JSONL file (thread-safe)."""
        with self._write_lock:  # Prevent concurrent writes
            with open(self.storage_path, 'a') as f:
                f.write(json.dumps(entry) + "\n")

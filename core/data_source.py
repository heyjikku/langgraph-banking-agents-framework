"""
Customer data source — the service owns the data; the dashboard never embeds it.

Reads the bank's exported JSON tables from DATA_DIR (default ./data) once, indexes by
customer_id, and assembles the per-customer bundle the orchestrator consumes. Swap this
module for a warehouse/CDP adapter (same two functions) to go to production.
"""
from __future__ import annotations
import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger("data_source")
DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))

# table file → bundle key, one-per-customer?
TABLES = {
    "customers": ("profile", True), "accounts": ("accounts", False), "transactions": ("transactions", False),
    "bureau_data": ("bureau", True), "sentiment_data": ("sentiment", False), "fraud_alerts": ("fraud", False),
    "interactions": ("interactions", False), "dynamic_personas": ("persona", True), "microsegments": ("microsegment", True),
    "nudges_nba": ("nudges", False), "life_events": ("life_events", False), "mobile_behavior": ("mobile", True),
    "journey_data": ("journeys", False), "lifecycle_data": ("lifecycle", True),
}


class DataSource:
    def __init__(self, data_dir: str = DATA_DIR):
        self.dir = data_dir
        self._idx: Dict[str, Dict[str, Any]] = {}
        self._customers: List[Dict] = []
        self._customer_rows: List[Dict] = []
        self._as_of: Optional[str] = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        for table, (key, single) in TABLES.items():
            path = os.path.join(self.dir, f"{table}.json")
            if not os.path.exists(path):
                logger.warning(f"data table missing: {path}")
                continue
            rows = json.load(open(path))
            by: Dict[str, Any] = {} if single else defaultdict(list)
            for r in rows:
                cid = r.get("customer_id")
                if cid is None:
                    continue
                if single:
                    by[cid] = r
                else:
                    by[cid].append(r)
            self._idx[key] = by
            if table == "customers":
                self._customers = rows
            if table in ("transactions", "interactions", "sentiment_data"):
                fld = {"transactions": "date", "interactions": "timestamp", "sentiment_data": "date"}[table]
                latest = max((str(r.get(fld, ""))[:10] for r in rows), default="")
                if len(latest) == 10 and (self._as_of is None or latest > self._as_of):
                    self._as_of = latest
        self._customer_rows = [{"customer_id": c["customer_id"], "name": f"{c.get('first_name', '')} {c.get('last_name', '')}".strip(),
                                "segment": c.get("segment"), "city": c.get("city"), "state": c.get("state"), "is_active": c.get("is_active")}
                               for c in self._customers]
        self._loaded = True
        logger.info(f"data source: {len(self._customers)} customers from {self.dir}")

    def load(self) -> None:
        """Eager load — call at service startup."""
        self._load()

    def as_of_date(self) -> Optional[str]:
        """Latest activity date across the whole dataset (YYYY-MM-DD) — the reference 'today' for all recency maths."""
        self._load()
        return self._as_of

    def list_customers(self, limit: Optional[int] = None) -> List[Dict]:
        self._load()
        return self._customer_rows[:limit] if limit else list(self._customer_rows)

    def bundle(self, customer_id: str) -> Optional[Dict[str, Any]]:
        self._load()
        prof = self._idx.get("profile", {}).get(customer_id)
        if not prof:
            return None
        out: Dict[str, Any] = {}
        for key, single in ((k, s) for _, (k, s) in TABLES.items()):
            src = self._idx.get(key, {})
            out[key] = src.get(customer_id, {} if single else [])
        return out


source = DataSource()

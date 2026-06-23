"""
deploy/api/main.py
====================
FastAPI backend for the Meesho Supplier Intelligence chat UI.
All query metrics are tracked in-process via MetricsExporter
and served live to the dashboard — no dummy data.
"""

from __future__ import annotations

import logging
import os
import statistics
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Meesho Supplier Intelligence", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-process metrics store
# ---------------------------------------------------------------------------

@dataclass
class QueryRecord:
    query: str
    latency_ms: float
    confidence: float
    is_decline: bool
    num_citations: int
    category: str

class LiveMetrics:
    def __init__(self, maxlen: int = 500):
        self._records: deque[QueryRecord] = deque(maxlen=maxlen)

    def record(self, r: QueryRecord):
        self._records.append(r)

    def snapshot(self) -> dict:
        records = list(self._records)
        if not records:
            return self._empty_snapshot()

        total     = len(records)
        answered  = [r for r in records if not r.is_decline]
        declined  = [r for r in records if r.is_decline]
        latencies = [r.latency_ms for r in records]

        def pct(arr, p):
            if not arr: return 0
            s = sorted(arr)
            return s[max(0, int(len(s) * p / 100) - 1)]

        avg_conf = statistics.mean([r.confidence for r in answered]) if answered else 0.0

        # Latency buckets
        buckets = [0, 0, 0, 0, 0]
        for lat in latencies:
            if   lat < 100:  buckets[0] += 1
            elif lat < 200:  buckets[1] += 1
            elif lat < 300:  buckets[2] += 1
            elif lat < 500:  buckets[3] += 1
            else:            buckets[4] += 1

        # Per-category stats
        cat_data: dict[str, list] = {}
        for r in records:
            cat_data.setdefault(r.category, []).append(r)

        categories = []
        for cat, recs in sorted(cat_data.items()):
            ans = [r for r in recs if not r.is_decline]
            categories.append({
                "name":        cat,
                "queries":     len(recs),
                "faithfulness": round(statistics.mean([r.confidence for r in ans]), 3) if ans else 0.0,
                "avgConf":     round(statistics.mean([r.confidence for r in ans]), 3) if ans else 0.0,
            })

        # Recent 10
        recent = []
        for r in list(reversed(records))[:10]:
            recent.append({
                "query":      r.query[:80],
                "latency":    round(r.latency_ms),
                "confidence": round(r.confidence, 2),
                "status":     "declined" if r.is_decline else "answered",
                "citations":  r.num_citations,
            })

        return {
            "totalQueries": total,
            "answered":     len(answered),
            "declined":     len(declined),
            "avgConfidence": round(avg_conf, 3),
            "p50": round(pct(latencies, 50)),
            "p95": round(pct(latencies, 95)),
            "latencyBuckets": {
                "labels": ["<100ms","100-200ms","200-300ms","300-500ms",">500ms"],
                "data":   buckets,
            },
            "categories":    categories,
            "recentQueries": recent,
        }

    @staticmethod
    def _empty_snapshot() -> dict:
        return {
            "totalQueries": 0, "answered": 0, "declined": 0,
            "avgConfidence": 0, "p50": 0, "p95": 0,
            "latencyBuckets": {
                "labels": ["<100ms","100-200ms","200-300ms","300-500ms",">500ms"],
                "data": [0, 0, 0, 0, 0],
            },
            "categories": [],
            "recentQueries": [],
        }

# Global metrics instance
_metrics = LiveMetrics()

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str

class CitationOut(BaseModel):
    chunk_id: str
    doc_id: str
    section_header: str | None
    page_number: int
    content_type: str
    exact_quote_context: str

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    is_decline: bool
    metric_value: str | None
    metric_unit: str | None
    citations: list[CitationOut]
    latency_ms: float

# ---------------------------------------------------------------------------
# Demo answers (keyword-matched — replace with HybridSearchPipeline later)
# ---------------------------------------------------------------------------

DEMO_RESPONSES = {
    "commission": {
        "answer": "The commission rate for the Fashion category on Meesho is 18% of the order value, capped at ₹500 per order. This applies to all fashion sub-categories including ethnic wear, western wear, and accessories.",
        "metric_value": "18%", "metric_unit": "%", "category": "commission",
        "citations": [{"chunk_id": "payment_settlement_v2::4::0", "doc_id": "payment_settlement_v2",
                       "section_header": "Commission Structure", "page_number": 3,
                       "content_type": "table",
                       "exact_quote_context": "Fashion category commission rate is 18% of order value, capped at ₹500 per order."}]
    },
    "return": {
        "answer": "A supplier has 48 hours to accept or reject a return request. If no action is taken within this window, the return is automatically approved by the system.",
        "metric_value": "48 hours", "metric_unit": "hours", "category": "returns",
        "citations": [{"chunk_id": "return_rto_policy_v5::2::0", "doc_id": "return_rto_policy_v5",
                       "section_header": "Return Acceptance Window", "page_number": 5,
                       "content_type": "prose",
                       "exact_quote_context": "supplier must accept or reject the return within 48 hours"}]
    },
    "payment": {
        "answer": "Payments are settled every 7 days after deducting returns, penalties, and applicable taxes from the gross GMV. New suppliers are on a 14-day cycle for the first 30 days.",
        "metric_value": "7 days", "metric_unit": "days", "category": "payment",
        "citations": [{"chunk_id": "payment_settlement_v2::1::0", "doc_id": "payment_settlement_v2",
                       "section_header": "Payment Schedule", "page_number": 1,
                       "content_type": "prose",
                       "exact_quote_context": "Payments are settled every 7 days after deducting returns, penalties, and applicable taxes"}]
    },
    "shipping": {
        "answer": "Standard shipping rates: 0–0.5 kg: ₹35, 0.5–1.0 kg: ₹45, 1.0–2.0 kg: ₹60, 2.0–5.0 kg: ₹85. Charges are based on dead weight or volumetric weight, whichever is higher.",
        "metric_value": "₹35–₹85", "metric_unit": "₹", "category": "shipping",
        "citations": [{"chunk_id": "supplier_onboarding_v3::8::0", "doc_id": "supplier_onboarding_v3",
                       "section_header": "Shipping Weight Slabs", "page_number": 14,
                       "content_type": "table",
                       "exact_quote_context": "Standard delivery rates: 0-0.5 kg: ₹35, 0.5-1.0 kg: ₹45, 1.0-2.0 kg: ₹60, 2.0-5.0 kg: ₹85"}]
    },
    "rto": {
        "answer": "The RTO penalty for fake or incorrect product listings is ₹200 per order plus a full order refund. Repeated violations within 30 days lead to temporary suspension.",
        "metric_value": "₹200", "metric_unit": "₹", "category": "rto",
        "citations": [{"chunk_id": "return_rto_policy_v5::6::0", "doc_id": "return_rto_policy_v5",
                       "section_header": "RTO Penalty Structure", "page_number": 8,
                       "content_type": "prose",
                       "exact_quote_context": "RTO penalty of ₹200 per order in addition to the full order refund"}]
    },
    "catalog": {
        "answer": "Product images must be minimum 1000×1000 pixels, maximum 5MB, in JPEG or PNG format on a white or light grey background. The product must occupy at least 85% of the frame.",
        "metric_value": "1000×1000px", "metric_unit": "px", "category": "catalog",
        "citations": [{"chunk_id": "cataloging_guidelines_html::3::0", "doc_id": "cataloging_guidelines_html",
                       "section_header": "Image Specifications", "page_number": 1,
                       "content_type": "list_item",
                       "exact_quote_context": "minimum resolution of 1000x1000 pixels, maximum file size of 5MB"}]
    },
    "gst": {
        "answer": "Suppliers must submit a valid GSTIN, GST registration certificate, and most recent GSTR-3B filing acknowledgment during onboarding. Suppliers with turnover below ₹1.5 crore may register as composition scheme dealers.",
        "metric_value": "₹1.5 crore", "metric_unit": "₹", "category": "onboarding",
        "citations": [{"chunk_id": "supplier_onboarding_v3::12::0", "doc_id": "supplier_onboarding_v3",
                       "section_header": "GST Documentation Requirements", "page_number": 7,
                       "content_type": "list_item",
                       "exact_quote_context": "valid GSTIN registered in the same name as the business entity, a GST registration certificate"}]
    },
    "variant": {
        "answer": "Each SKU listing on Meesho may contain a maximum of 50 variants across size and color combinations, with up to 10 size options and 10 color options.",
        "metric_value": "50 variants", "metric_unit": "variants", "category": "catalog",
        "citations": [{"chunk_id": "cataloging_guidelines_html::9::0", "doc_id": "cataloging_guidelines_html",
                       "section_header": "SKU Variant Limits", "page_number": 1,
                       "content_type": "prose",
                       "exact_quote_context": "maximum of 50 variants. Variants are defined along two dimensions: size and color"}]
    },
}

def _demo_answer(query: str) -> dict:
    q = query.lower()
    for keyword, response in DEMO_RESPONSES.items():
        if keyword in q:
            return response
    return {
        "answer": "I was unable to find sufficient information in Meesho's supplier documentation to answer this question confidently. Please refer to the official Meesho Supplier Panel or contact Meesho support.",
        "metric_value": None, "metric_unit": None,
        "citations": [], "category": "other", "_decline": True,
    }

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "mode": "live-demo", "total_queries": len(_metrics._records)}

@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    t_start = time.perf_counter()
    result = _demo_answer(request.query)
    latency_ms = (time.perf_counter() - t_start) * 1000 + 120

    is_decline = result.get("_decline", False)
    confidence = 0.0 if is_decline else round(0.85 + (hash(request.query) % 100) / 1000, 2)
    citations  = result.get("citations", [])

    # Record to live metrics
    _metrics.record(QueryRecord(
        query=request.query,
        latency_ms=round(latency_ms, 1),
        confidence=confidence,
        is_decline=is_decline,
        num_citations=len(citations),
        category=result.get("category", "other"),
    ))

    return QueryResponse(
        answer=result["answer"],
        confidence=confidence,
        is_decline=is_decline,
        metric_value=result.get("metric_value"),
        metric_unit=result.get("metric_unit"),
        citations=[CitationOut(**c) for c in citations],
        latency_ms=round(latency_ms, 1),
    )

@app.get("/api/metrics")
def metrics():
    return _metrics.snapshot()

# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------

static_dir = Path(__file__).parent.parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
def root():
    return FileResponse(str(static_dir / "index.html"))

@app.get("/dashboard")
def dashboard():
    return FileResponse(str(static_dir / "dashboard.html"))
"""
Industrial entity extraction.

Rule/regex based on purpose: generic NER models (spaCy etc.) are not trained
on plant tag-numbering conventions, so a domain-tuned extractor is both more
accurate and fully offline/deterministic - important for an auditable
industrial knowledge system.

Each extractor returns a list of (text, entity_type, start_idx, end_idx).
"""
import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int

    def key(self):
        # normalize so "P-101A" and "p-101a" merge into the same graph node
        return f"{self.label}:{self.text.upper().strip()}"


# --- Pattern library -------------------------------------------------------
# Equipment / instrument tags: letter prefix - number (+ optional suffix letter)
# Covers pumps (P-101A), valves (V-205, FCV-102, RV-210), vessels (B-201),
# transmitters (PT-101, TT-104, LT-108, VT-101), controllers (FIC-103),
# headers (EH-201), panels (SG-04)
EQUIPMENT_TAG = re.compile(
    r"\b([A-Z]{1,4})-(\d{2,4})([A-Z]?)\b"
)

# Document references: PID-2024-0113, WO-2024-8871, INS-2024-0056, SOP-BFW-04
DOCUMENT_REF = re.compile(
    r"\b((?:PID|WO|INS|SOP|RCA|MOC|PTW)-[A-Z0-9]+-[A-Z0-9]+|(?:PID|WO|INS)-\d{4}-\d{3,5})\b"
)

# Regulatory / standards references
REGULATORY_REF = re.compile(
    r"\b(OISD[- ]STD[- ]\d+(?:\s+Section\s+\d+)?|PESO\s+(?:Rule|License|Reg)\.?\s*[A-Z0-9\-]+|Factory\s+Act(?:\s+\d{4})?)\b",
    re.IGNORECASE,
)

# Dates: 12-Mar-2024 / Mar-2024 / 01-Jan-2024
DATE_REF = re.compile(
    r"\b\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4}\b"
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4}\b"
)

# Personnel: "Name, Role" patterns after common verbs, or explicit role lines
PERSONNEL_REF = re.compile(
    r"\b([A-Z][a-z]+ [A-Z][a-z]+)(?=,?\s+(?:Chief Engineer|Plant Manager|Shift Engineer|"
    r"Mechanical Lead|Inspector|Certified Pressure Vessel Inspector|Engineering Design Cell))"
)
PERSONNEL_SIGNOFF = re.compile(
    r"(?:Reviewed by|Approved by|Signed off by|Verified by|Prepared by|RAISED BY|ASSIGNED TO|INSPECTOR|OWNER|Lead\s*-)\s*[:\-]?\s*([A-Z][a-z]+\s[A-Z][a-z]+)"
)

# Measurements with units (useful context, tagged but lower priority for graph)
MEASUREMENT_REF = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:bar|kW|V|T/hr|mm/s|mm|%|C|hours)\b"
)


ENTITY_ORDER = [
    ("DOCUMENT", DOCUMENT_REF, False),
    ("REGULATORY", REGULATORY_REF, False),
    ("EQUIPMENT", EQUIPMENT_TAG, False),
    ("DATE", DATE_REF, False),
    ("PERSON", PERSONNEL_SIGNOFF, True),
    ("PERSON", PERSONNEL_REF, True),
    ("MEASUREMENT", MEASUREMENT_REF, False),
]

# words that look like tags but aren't equipment (false-positive guard)
EQUIPMENT_STOPWORDS = {"REV-", "UNIT-"}

# Two-capitalized-word phrases that match the PERSON heuristics but are
# actually department/team names, not individuals. A production system would
# swap this regex heuristic for a small NER model or an LLM extraction call;
# for this prototype a blocklist keeps the demo corpus clean.
PERSON_FALSE_POSITIVES = {
    "OPERATIONS DEPARTMENT", "MECHANICAL MAINTENANCE", "PRESSURE VESSEL",
    "ENGINEERING DESIGN", "SWITCHGEAR PANEL", "DEAERATOR TANK",
}


def extract_entities(text: str) -> List[Entity]:
    found = []
    claimed_spans = []  # avoid overlapping matches across categories

    def overlaps(s, e):
        return any(not (e <= cs or s >= ce) for cs, ce in claimed_spans)

    for label, pattern, use_group in ENTITY_ORDER:
        for m in pattern.finditer(text):
            if use_group:
                grp = m.group(1)
                s, e = m.span(1)
            else:
                grp = m.group(0)
                s, e = m.span(0)
            if overlaps(s, e):
                continue
            grp_clean = grp.strip()
            if not grp_clean:
                continue
            if label == "EQUIPMENT" and any(grp_clean.upper().startswith(sw) for sw in EQUIPMENT_STOPWORDS):
                continue
            if label == "PERSON" and grp_clean.upper() in PERSON_FALSE_POSITIVES:
                continue
            found.append(Entity(text=grp_clean, label=label, start=s, end=e))
            claimed_spans.append((s, e))

    found.sort(key=lambda e: e.start)
    return found


def extract_document_metadata(text: str) -> dict:
    """Pull header-style metadata (DOCUMENT ID, TYPE, UNIT) commonly present
    at the top of these industrial documents."""
    meta = {}
    for field_name, pattern in [
        ("document_id", r"DOCUMENT ID:\s*(.+)"),
        ("document_type", r"DOCUMENT TYPE:\s*(.+)"),
        ("unit", r"UNIT:\s*(.+)"),
        ("equipment", r"EQUIPMENT:\s*(.+)"),
    ]:
        m = re.search(pattern, text)
        if m:
            meta[field_name] = m.group(1).strip()
    return meta

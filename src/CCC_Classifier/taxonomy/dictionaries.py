
# -*- coding: utf-8 -*-
"""
Canonical taxonomy dictionaries for CCC classifier.

Design choices (as requested):
- Keep "No Customer Input"
- Remove:
  - "Unclassified Domain"
  - "Unclassified Subdomain"
  - "Unclassified Issue Origin"
  - "Unclassified Cause"
  - "Unclassified Driver"
- Anything outside canonical labels should be handled as: "Other: <free text>"
  (implemented in taxonomy/canon.py and pipeline stages, not here).
"""

from __future__ import annotations

from typing import Dict, List

EXPECTED_KEYS_ORDER: List[str] = [
    "contact_type",
    "domain",
    "subdomain",
    "issue_origin",
    "SHORT_SUMMARY",
    "DETAILED_SUMMARY",
    "confidence",
]

# -------------------------
# Contact Types (canonical)
# -------------------------
CONTACT_TYPES_CANON: List[str] = ["Action Requested", "Information Requested", "Unclear Contact"]

# -------------------------
# Domains (canonical)
# NOTE: "Unclassified Domain" removed by request.
# -------------------------
DOMAINS_CANON: List[str] = [
    "Product",
    "Billing",
    "Order & Fulfillment",
    "Returns",
    "Technical Support",
    "Programs & Rewards",
    "Case Management",
    "Policy & Compliance",
    "No Customer Input",
    "Customer Feedback",
]

# --------------------------------
# Subdomains (canonical) - flat list
# --------------------------------
SUBDOMAINS_CANON: List[str] = [
    "Delivery delay / late delivery",
    "Missing order / missing tote",
    "Missing narcotics",
    "Order status / ETA follow-up",
    "Warehouse or route disruption",
    "Vaccine availability (COVID / Flu)",
    "Vaccine allocation issues",
    "Exception order requests (MOH)",
    "Unable to order in PharmaClick",
    "Portal access / login / password reset",
    "System / PharmaClick display issues",
    "Returns – damaged product",
    "Returns – short shipment",
    "Returns – incorrect item",
    "Billing / invoice discrepancy",
    "Credit follow-up",
    "Backorder inquiry",
    "Out-of-stock medication",
    "Manual order request",
    "Controlled substance handling",
    "Missing documentation / paperwork",
    "Consumables / supplies ordering",
    "General order placement assistance",
    "Escalation / supervisor follow-up",
]

# --------------------------------
# Issue Origin (canonical)
# --------------------------------
ISSUE_ORIGINS_CANON: List[str] = [
    "Transportation",
    "Weather",
    "Distribution Center (DC)",
    "Supplier/Manufacturer",
    "Carrier",
    "Warehouse/Route",
    "System/Portal",
    "Customer Internet/Network",
    "Customer"
]

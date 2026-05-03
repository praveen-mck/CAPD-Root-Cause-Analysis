
# -*- coding: utf-8 -*-
"""
Canonical taxonomy dictionaries for CCC classifier.

Design choices (as requested):
- Keep "No Customer Input"
- Remove:
  - "Unclassified Domain"
  - "Unclassified Subdomain"
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
    "SHORT_SUMMARY",
    "DETAILED_SUMMARY",
    "confidence",
]

# -------------------------
# Contact Types (canonical)
# -------------------------
CONTACT_TYPES_CANON: List[str] = ["Issue", "Inquiry", "Request", "Unclear Contact"]

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
# Subdomains per Domain (canonical)
# NOTE: "Unclassified Subdomain" removed from every domain by request.
# NOTE: Entire "Unclassified Domain" key removed by request.
# --------------------------------




SUBDOMAINS_BY_DOMAIN_CANON: Dict[str, List[str]] = {
    "Product": [
        "Product Availability",
        "Product Expiry Information",
        "Product Details & Specifications",
        "Product Quality Issue",
        "Product Recall",
    ],
    "Billing": [
        "Invoice Request",
        "Invoice Discrepancy",
        "Account Hold / AR Hold",
        "Pricing & Quotes",
        "Credits & Adjustments",
        "Reports & Statements",
        "Payment Assistance",
        "Invoice code meanings",
    ],
    "Order & Fulfillment": [
        "Order Placement",
        "Order Status",
        "Delivery Status / ETA",
        "Shipment Shortage",
        "Incorrect Shipment Item",
        "Delivery Documentation",
        "Order cancellation",
        "Order modification",
        "Delivery closure notice",
        "Packaging return policy",
        "Holiday closure notice",
        "Delivery schedule update",
        "Delivery hold request",
        "Delivery equipment issue",
        "Price stickers",
    ],
    "Returns": [
        "Return Eligibility",
        "Return Authorization (RMA)",
        "Return Documentation",
        "Return Status",
        "Return Defect",
        "Return packaging",
        "Return packaging instructions",
    ],
    "Technical Support": [
        "Authentication Issue",
        "Portal Error",
        "System Outage",
        "Account Administration",
        "Invoice reupload",
    ],
    "Programs & Rewards": [
        "Publicly Funded Campaign Info",
        "Rewards Cards Request",
        "Rebate Information",
        "Promotion code issue",
        "Promotional items availability"
    ],
    "Case Management": [
        "Callback Request",
        "Case Status Update",
        "Case Closure Request",
        "Reconnect with Agent",
        "Contact Information Request",
        "Account merge request",
        "Account access change",
        "Temporary closure notice",
        "Submit proof of delivery",
    ],
    "Policy & Compliance": [
        "Regulatory Documentation",
        "Service Closures & Notices",
        # this is a subdomain - can't be a root cause
        #"Controlled substance policy",
        "Access removal request",
    ],
    "Customer Feedback": [
        "Appreciation",
    ],
    "No Customer Input": [
        "No Customer Input",
    ],
}

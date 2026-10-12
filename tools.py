"""tools.py - Task 6: check_order_status + designed escalation_score."""
from dataset import ORDERS

# Formula: escalation_score = 0.6 * delayed_shipment + 0.4 * (days_since_created / 30)
# delayed_shipment 0 ya 1 hai, age_norm 0 se 1 ke beech, to score hamesha [0, 1] mein rehta hai.
# Purana aur delayed order zyada urgent hota hai, isliye age ko score mein jod rahe hain.
W_DELAY = 0.6
W_AGE = 0.4
MAX_DAYS = 30


def _score(order):
    age_norm = order["days_since_created"] / MAX_DAYS
    return round(W_DELAY * (1 if order["delayed_shipment"] else 0) + W_AGE * age_norm, 4)


def _percentile(values, p):
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


_ALL_SCORES = [_score(o) for o in ORDERS]
# Threshold = apne dataset ke scores ka 80th percentile
ESCALATION_THRESHOLD = round(_percentile(_ALL_SCORES, 80), 3)
_INDEX = {o["record_id"]: o for o in ORDERS}


def check_order_status(record_id: str) -> dict:
    """Look up an order by its record ID (for example NYK-001).

    Returns the order's status, order value in INR, an escalation score between
    0 and 1 (combining delayed shipment and order age), and whether escalation
    to a human agent is recommended. If the ID does not exist, returns found=False.
    """
    order = _INDEX.get(str(record_id).strip().upper())
    if order is None:
        return {"found": False, "record_id": record_id, "error": "Order not found"}
    score = _score(order)
    return {
        "found": True,
        "record_id": order["record_id"],
        "status": order["status"],
        "order_value_inr": order["order_value_inr"],
        "escalation_score": score,
        "escalation_recommended": score >= ESCALATION_THRESHOLD,
    }


if __name__ == "__main__":
    print("Formula: escalation_score = 0.6*delayed_shipment + 0.4*(days_since_created/30)")
    print("Score min/max in dataset:", min(_ALL_SCORES), max(_ALL_SCORES))
    for p in (50, 80, 90):
        print(f"{p}th percentile of scores: {_percentile(_ALL_SCORES, p):.3f}")
    n_esc = sum(s >= ESCALATION_THRESHOLD for s in _ALL_SCORES)
    print(f"Chosen threshold (80th percentile) = {ESCALATION_THRESHOLD}")
    print(f"Orders at/above threshold: {n_esc} of {len(ORDERS)}")
    for rid in ("NYK-001", "NYK-010", "NYK-999"):
        print(rid, "->", check_order_status(rid))
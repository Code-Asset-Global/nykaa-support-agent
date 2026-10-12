"""dataset.py - Nykaa order dataset (seeded, deterministic)."""
import random
from collections import Counter

CATEGORIES = {"Apparel": 12, "Beauty": 12, "Footwear": 10, "Home": 8, "Electronics": 8}
STATUSES = {"Placed": 8, "Shipped": 10, "Delivered": 20, "Returned": 7, "Refunded": 5}

# Price range (INR): vaires and depend on online-order range
PRICE_RANGE_INR = {
    "Beauty": (300, 3000),
    "Apparel": (500, 5000),
    "Footwear": (800, 6000),
    "Home": (700, 15000),
    "Electronics": (1500, 60000),
}
DELAY_PROB = 0.20
START_SEED = 42


def _build(seed):
    rng = random.Random(seed)
    cats = [c for c, n in CATEGORIES.items() for _ in range(n)]
    stats = [s for s, n in STATUSES.items() for _ in range(n)]
    rng.shuffle(cats)
    rng.shuffle(stats)
    orders = []
    for i, (cat, status) in enumerate(zip(cats, stats), start=1):
        lo, hi = PRICE_RANGE_INR[cat]
        orders.append({
            "record_id": f"NYK-{i:03d}",
            "category": cat,
            "status": status,
            "order_value_inr": rng.randint(lo, hi),
            "days_since_created": rng.randint(0, 30),
            "delayed_shipment": rng.random() < DELAY_PROB,
        })
    return orders


def generate_orders():
    seed = START_SEED
    while True:
        orders = _build(seed)
        pct = 100 * sum(o["delayed_shipment"] for o in orders) / len(orders)
        if 10 <= pct <= 30:
            return orders, seed
        seed += 1


ORDERS, SEED_USED = generate_orders()

if __name__ == "__main__":
    print("Total records:", len(ORDERS))
    print("Seed used:", SEED_USED)
    print("Count per category:", dict(Counter(o["category"] for o in ORDERS)))
    print("Count per status:", dict(Counter(o["status"] for o in ORDERS)))
    pct = 100 * sum(o["delayed_shipment"] for o in ORDERS) / len(ORDERS)
    print(f"delayed_shipment True: {pct:.1f}%")
    print("Sample:", ORDERS[0])
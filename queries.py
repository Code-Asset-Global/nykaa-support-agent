"""queries.py - Task 4 aur Task 5 ke queries (calibration alag, demo alag)."""

# Threshold tay karne ke liye
CALIB_IN_SCOPE = [
    "What is the return period for lipstick?",
    "How long does a bank refund take for cash on delivery?",
    "Can I cancel my order after it has shipped?",
    "How many loyalty points do I need to redeem?",
    "What is the warranty on a laptop?",
    "What should I do if my parcel arrives broken?",
]
CALIB_OUT_OF_SCOPE = [
    "Who won the cricket world cup in 2011?",
    "How do I bake a chocolate cake?",
    "What is the capital of France?",
    "Explain how a car engine works.",
]

# Demo (Task 4) aur Precision/Recall (Task 5): (query, sahi doc_ids)
DEMO_QUERIES = [
    ("How many days do I have to return a beauty product?", ["return_window"]),
    ("When will I get my refund for a COD order?", ["cod_refund"]),
    ("Can I exchange shoes for a different size?", ["size_exchange"]),
    ("What should I do if the item I received is damaged?", ["damaged_item_claim"]),
    ("Who do I contact if my shipment is delayed by more than a week?", ["escalation_matrix", "delivery_sla"]),
]
OUT_OF_SCOPE_DEMO = "What is the best way to learn guitar?"
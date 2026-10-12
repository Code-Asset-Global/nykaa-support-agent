"""kb.py - Nykaa support knowledge base (12 documents).
Note: ye policies is project ke liye banayi gayi fictional policies hain."""
import os
import re

KB_DOCUMENTS = [
    {"doc_id": "return_window", "topic": "Return window by product category",
     "text": "Nykaa accepts returns within a fixed window that depends on the product category. "
             "Beauty products can be returned within 7 days of delivery, and only if the seal is unbroken. "
             "Apparel and Footwear can be returned within 15 days if the tags are attached and the item is unworn. "
             "Electronics and Home items have a 10-day return window and need the original packaging."},
    {"doc_id": "cod_refund", "topic": "COD refund timelines",
     "text": "For Cash on Delivery orders, refunds cannot go back to a card because no online payment was made. "
             "The customer must share bank account details or choose wallet credit instead. "
             "Bank refunds are processed within 5 to 7 working days after the returned item passes the quality check. "
             "Wallet credit is added within 24 hours of the quality check."},
    {"doc_id": "delivery_sla", "topic": "Delivery SLAs",
     "text": "Standard delivery in metro cities takes 2 to 4 working days. "
             "Non-metro and remote pin codes take 5 to 8 working days. "
             "Express delivery, where available, reaches select cities within 24 hours of dispatch for an extra fee. "
             "An order is marked as delayed when it crosses the promised delivery date by more than 2 days."},
    {"doc_id": "reverse_pickup", "topic": "Reverse-pickup eligibility",
     "text": "Reverse pickup is available for returns and exchanges at pin codes served by courier partners. "
             "The pickup is scheduled within 2 working days after a return request is approved. "
             "Items must be packed in the original packaging with all tags and accessories. "
             "If a pin code is not serviceable, the customer can drop the item at a partner courier center and upload the receipt."},
    {"doc_id": "warranty_terms", "topic": "Warranty terms by category",
     "text": "Electronics carry a 12-month manufacturer warranty from the delivery date, covering manufacturing defects only. "
             "Home appliances have a 6-month warranty, while furniture and decor have 3 months. "
             "Apparel, Footwear and Beauty products have no warranty, but defects can be reported under the damaged-item claim process. "
             "Physical damage, water damage and unauthorized repairs void the warranty."},
    {"doc_id": "order_cancellation", "topic": "Order-cancellation policy",
     "text": "An order can be cancelled for free until it is shipped. "
             "Once an order is in Shipped status, cancellation is not possible and the customer must refuse delivery or request a return after receiving it. "
             "Prepaid cancellations are refunded to the original payment method within 5 to 7 working days. "
             "Cancelling more than 3 COD orders in a month may lead to COD being disabled on the account."},
    {"doc_id": "loyalty_points", "topic": "Loyalty-points redemption policy",
     "text": "Customers earn 1 loyalty point for every 100 rupees spent on delivered orders. "
             "Each point is worth 1 rupee, and at least 200 points are needed to redeem. "
             "Points can cover up to 20 percent of the order value on a single order. "
             "Points expire 12 months after they are earned, and points from returned orders are deducted."},
    {"doc_id": "payment_failure", "topic": "Payment-failure and retry policy",
     "text": "If a payment fails but the money is deducted, the amount is automatically reversed within 5 to 7 working days. "
             "Customers can retry payment up to 3 times for the same order within 30 minutes. "
             "After that, the order is cancelled automatically and the cart is restored. "
             "Switching to another payment method, such as UPI or a different card, often resolves repeated failures."},
    {"doc_id": "size_exchange", "topic": "Size-exchange policy",
     "text": "Size exchange is available for Apparel and Footwear once per order item within 15 days of delivery. "
             "The replacement size is shipped only after the original item is picked up and passes the quality check, unless the customer pays a refundable deposit. "
             "If the requested size is out of stock, a refund or wallet credit is offered instead. "
             "Exchanges are not available for innerwear or items marked as final sale."},
    {"doc_id": "damaged_item_claim", "topic": "Damaged-item claim process",
     "text": "If an item arrives damaged or incorrect, the customer must report it within 48 hours of delivery. "
             "They need to share clear photos of the item and the outer packaging, plus an unboxing video for orders above 5000 rupees. "
             "After verification, which takes up to 2 working days, a replacement or a full refund is issued. "
             "Reverse pickup for damaged items is free of charge."},
    {"doc_id": "international_shipping", "topic": "International shipping restrictions",
     "text": "Nykaa ships internationally only to a limited list of countries, such as the UAE, Singapore and the USA. "
             "Aerosols, high-alcohol perfumes and batteries cannot be shipped internationally because of aviation rules. "
             "Customs duties and import taxes are paid by the customer and are not refundable. "
             "International orders cannot use Cash on Delivery and take 7 to 15 working days to arrive."},
    {"doc_id": "escalation_matrix", "topic": "Customer-support escalation matrix",
     "text": "Level 1 is the frontline support agent, who responds to queries within 24 hours. "
             "If a case is unresolved after 24 hours, it moves to Level 2, the team lead, who responds within 48 hours. "
             "Level 3 is the customer experience manager, who handles cases open for more than 3 days, shipments delayed more than 7 days, or refunds above 10000 rupees. "
             "Fraud or safety complaints skip directly to Level 3."},
]


def count_sentences(text):
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


def export_to_files(folder="kb"):
    os.makedirs(folder, exist_ok=True)
    for d in KB_DOCUMENTS:
        with open(os.path.join(folder, d["doc_id"] + ".txt"), "w") as f:
            f.write(d["text"])


if __name__ == "__main__":
    print("Total documents:", len(KB_DOCUMENTS))
    for d in KB_DOCUMENTS:
        n = count_sentences(d["text"])
        print(f"{d['doc_id']:24} sentences={n}  ok={2 <= n <= 5}")
    export_to_files()
    print("Files written to kb/")
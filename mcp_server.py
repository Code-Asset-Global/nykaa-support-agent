"""mcp_server.py - Task 14: check_order_status ko MCP tool ke roop mein expose karo."""
from fastmcp import FastMCP
from tools import check_order_status as _check_order_status

mcp = FastMCP("Nykaa Order Tools")


@mcp.tool()
def check_order_status(record_id: str) -> dict:
    """Look up a Nykaa order by its record ID (for example NYK-001).

    Returns the order status, order value in INR, an escalation score between
    0 and 1 (combining delayed shipment and order age), and whether escalation
    to a human agent is recommended. If the ID does not exist, found=False is returned.
    """
    return _check_order_status(record_id)


if __name__ == "__main__":
    # HTTP transport, endpoint: http://127.0.0.1:8000/mcp
    mcp.run(transport="http", host="127.0.0.1", port=8000)
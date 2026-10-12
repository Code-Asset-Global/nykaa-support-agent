"""mcp_client.py - Task 14: alag MCP client jo server ko call karta hai."""
import asyncio
from fastmcp import Client

URL = "http://127.0.0.1:8000/mcp"


async def main():
    async with Client(URL) as client:
        tools = await client.list_tools()
        print("Tools on server:", [t.name for t in tools])
        for rid in ("NYK-001", "NYK-010", "NYK-999"):
            res = await client.call_tool("check_order_status", {"record_id": rid})
            print(f"\n--- call_tool(check_order_status, record_id={rid}) ---")
            for block in getattr(res, "content", res):
                print("MCP content:", getattr(block, "text", block))
            print("structured:", getattr(res, "structured_content", None))


asyncio.run(main())
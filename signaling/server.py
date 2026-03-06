from __future__ import annotations

"""
Minimal WebRTC signaling server.

Serves the static WebRTC page and relays SDP/ICE messages between
sender and receiver via WebSocket.
"""

import json
import logging
from aiohttp import web, WSMsgType

from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Room state: room_id -> {"sender": ws, "receiver": ws}
rooms: dict[str, dict[str, web.WebSocketResponse]] = {}


STATIC_DIR = Path(__file__).parent / "static" if (Path(__file__).parent / "static").exists() else Path("static")


async def index_handler(request: web.Request) -> web.FileResponse:
    """Serve index.html for the root path (with any query parameters)."""
    return web.FileResponse(STATIC_DIR / "index.html")


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """Handle WebSocket connections for SDP/ICE signaling."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    role = request.query.get("role", "unknown")
    room_id = request.query.get("room", "default")

    # Register this peer in the room
    if room_id not in rooms:
        rooms[room_id] = {}
    rooms[room_id][role] = ws
    logger.info(f"[{room_id}] {role} connected")

    # Notify the peer that the other side is already here (if it is)
    other_role = "receiver" if role == "sender" else "sender"
    other_ws = rooms[room_id].get(other_role)
    if other_ws and not other_ws.closed:
        # Tell both sides the other is ready
        await ws.send_json({"type": "peer-joined", "role": other_role})
        await other_ws.send_json({"type": "peer-joined", "role": role})

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                # Forward message to the other peer in the same room
                target_ws = rooms.get(room_id, {}).get(other_role)
                if target_ws and not target_ws.closed:
                    await target_ws.send_json(data)
                else:
                    logger.warning(f"[{room_id}] No {other_role} to relay to")
            elif msg.type == WSMsgType.ERROR:
                logger.error(f"[{room_id}] WebSocket error from {role}: {ws.exception()}")
    finally:
        # Clean up on disconnect
        if room_id in rooms:
            rooms[room_id].pop(role, None)
            if not rooms[room_id]:
                del rooms[room_id]
        logger.info(f"[{room_id}] {role} disconnected")

    return ws


async def health_handler(request: web.Request) -> web.Response:
    """Health check endpoint."""
    return web.json_response({"status": "ok", "rooms": len(rooms)})


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_get("/health", health_handler)
    # Serve other static assets (CSS, JS, etc.) if any are added later
    app.router.add_static("/static", str(STATIC_DIR))
    return app


if __name__ == "__main__":
    app = create_app()
    logger.info("Starting signaling server on 0.0.0.0:8080")
    web.run_app(app, host="0.0.0.0", port=8080)

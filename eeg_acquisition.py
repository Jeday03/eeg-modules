"""
eeg_acquisition.py — EEG Publisher (BrainFlow -> WebSocket JSON)

This script starts a BrainFlow session, collects EEG samples in "chunks", and publishes
those chunks over WebSocket to any connected clients.

High-level architecture
-----------------------
1) BrainFlow:
   - Initializes BoardShim with a board_id (default: SYNTHETIC_BOARD).
   - Prepares the session and starts streaming.

2) Web server (aiohttp):
   - Runs an HTTP server with a WebSocket endpoint at: GET /ws
   - Keeps a set of connected WebSocket clients.

3) Publishing loop:
   - The async task `broadcast()` runs continuously.
   - Every interval (defined by --hz), it reads the latest N samples from the board
     (`get_current_board_data(n)`) and sends them to all connected clients.

Usage examples
--------------
1) Run with the synthetic board (default), port 8080:
   python eeg_acquisition.py

2) Run with a custom publish rate and chunk size:
   python eeg_acquisition.py --hz 20 --n 50

3) Run with a real board (example using a serial port):
   python eeg_acquisition.py --board <BOARD_ID> --serial COM3

CLI parameters
--------------
--board : int
    BrainFlow Board ID. Default: SYNTHETIC_BOARD.
--serial : str
    Serial port (when applicable).
--port : int
    HTTP server port.
--hz : float
    Approximate publish frequency (messages/second).
--n : int
    Number of samples per chunk read via get_current_board_data(n).

Message format (JSON)
---------------------
The server sends text messages (JSON string) shaped like:

{
  "type": "eeg_chunk",
  "fs": <float|int>,          # board sampling rate (Hz)
  "ch_names": [<str>...],     # EEG channel identifiers (here: indices as strings)
  "ts": [<float>...],         # timestamps (seconds) for each sample
  "eeg": [                    # list per channel
    [<float>...],             # channel 0 (n samples)
    [<float>...],             # channel 1
    ...
  ]
}

Important notes
---------------
- get_current_board_data(n) returns the "latest n samples currently available" in the
  internal ring buffer, which can cause overlap between consecutive chunks if the publish
  interval is short and/or if production/consumption rates differ.
- This script does not process inbound client messages; it only keeps the socket open
  and pushes data out.
"""

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from aiohttp import web
import asyncio
import json
import argparse

parser = argparse.ArgumentParser(description="EEG Publisher (BrainFlow -> WebSocket JSON)")
parser.add_argument("--board", type=int, default=BoardIds.SYNTHETIC_BOARD.value)
parser.add_argument("--serial", type=str, default="")
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--hz", type=float, default=10.0)
parser.add_argument("--n", type=int, default=10)
args = parser.parse_args()

# ----------------------------
# BrainFlow initialization
# ----------------------------
params = BrainFlowInputParams()
if args.serial:
    params.serial_port = args.serial

board_id = args.board
board = BoardShim(board_id, params)
board.prepare_session()
board.start_stream()

# Set of currently connected WebSocket clients
clients = set()


async def broadcast() -> None:
    """
    Continuous loop that publishes EEG data to WebSocket clients.

    - Defines the publish interval from `--hz`.
    - Retrieves EEG channel indices and timestamp channel index from the board.
    - On each iteration:
        * reads the latest N samples (args.n)
        * builds a JSON message (as text)
        * sends it to all connected clients
        * drops dead connections
    """
    interval = 1.0 / max(args.hz, 0.1)

    eeg_ch = board.get_eeg_channels(board_id)
    ts_ch = board.get_timestamp_channel(board_id)

    fs = BoardShim.get_sampling_rate(board_id)
    ch_names = [str(i) for i in eeg_ch]

    while True:
        data = board.get_current_board_data(args.n)

        # data shape: [n_total_channels, n_samples]
        if data.shape[1] > 0:
            eeg = [data[i].tolist() for i in eeg_ch]
            ts = data[ts_ch].tolist()

            msg = json.dumps(
                {
                    "type": "eeg_chunk",
                    "fs": fs,
                    "ch_names": ch_names,
                    "ts": ts,
                    "eeg": eeg,
                }
            )

            dead = []
            for ws in clients:
                try:
                    await ws.send_str(msg)
                except Exception:
                    dead.append(ws)

            for ws in dead:
                clients.discard(ws)

        await asyncio.sleep(interval)


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    """
    WebSocket endpoint handler (GET /ws).

    - Accepts and prepares a WebSocket connection.
    - Adds the socket to the global `clients` set.
    - Keeps the connection open by consuming inbound messages (ignored).
    - Removes the socket from the set when the connection ends.

    Returns:
        web.WebSocketResponse: the prepared WebSocket connection.
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    clients.add(ws)

    try:
        async for _ in ws:
            # This server does not process inbound client messages.
            pass
    finally:
        clients.discard(ws)

    return ws


async def on_startup(app: web.Application) -> None:
    """aiohttp startup hook: create the async publishing task."""
    app["task"] = asyncio.create_task(broadcast())


async def on_cleanup(app: web.Application) -> None:
    """
    aiohttp cleanup hook: stop the publishing task and close BrainFlow session.

    - Cancels the `broadcast` task.
    - Attempts to stop streaming and release the BrainFlow session.
    """
    app["task"].cancel()
    try:
        board.stop_stream()
        board.release_session()
    except Exception:
        pass


# ----------------------------
# aiohttp app
# ----------------------------
app = web.Application()
app.router.add_get("/ws", ws_handler)
app.on_startup.append(on_startup)
app.on_cleanup.append(on_cleanup)

web.run_app(app, port=args.port)

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


params = BrainFlowInputParams()
if args.serial:
    params.serial_port = args.serial

board_id = args.board
board = BoardShim(board_id, params)
board.prepare_session()
board.start_stream()


clients = set()


async def broadcast() -> None:
    interval = 1.0 / max(args.hz, 0.1)

    eeg_ch = board.get_eeg_channels(board_id)
    ts_ch = board.get_timestamp_channel(board_id)

    fs = BoardShim.get_sampling_rate(board_id)
    ch_names = [str(i) for i in eeg_ch]

    while True:
        data = board.get_current_board_data(args.n)

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
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    clients.add(ws)

    try:
        async for _ in ws:
            pass
    finally:
        clients.discard(ws)

    return ws


async def on_startup(app: web.Application) -> None:
    app["task"] = asyncio.create_task(broadcast())


async def on_cleanup(app: web.Application) -> None:
    app["task"].cancel()
    try:
        board.stop_stream()
        board.release_session()
    except Exception:
        pass



app = web.Application()
app.router.add_get("/ws", ws_handler)
app.on_startup.append(on_startup)
app.on_cleanup.append(on_cleanup)

web.run_app(app, port=args.port)

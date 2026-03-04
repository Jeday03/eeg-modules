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
    """
    This coroutine continuously pulls the most recent EEG samples from a BrainFlow
    board buffer and broadcasts them as JSON messages to all connected WebSocket
    clients.

    The coroutine is responsible for:
    - Computing the broadcast interval based on the user-defined publishing rate.
    - Discovering EEG and timestamp channel indices from the selected board.
    - Fetching the most recent `n` samples from the board internal ring buffer.
    - Packing EEG and timestamps into a JSON payload and sending it to all clients.
    - Detecting disconnected clients and removing them from the active set.

    The message schema sent to clients is:

    - type: str
        Always "eeg_chunk".
    - fs: int
        Sampling rate (Hz) reported by BrainFlow for the chosen board.
    - ch_names: list[str]
        Channel identifiers (stringified EEG channel indices from BrainFlow).
    - ts: list[float]
        Timestamps for each sample.
    - eeg: list[list[float]]
        EEG samples grouped by channel (channels × samples).

    Parameters
    ----------
    None
        This coroutine uses global runtime configuration via `args`, `board_id`,
        `board`, and the global `clients` set.

    Returns
    -------
    None
        This coroutine runs indefinitely until the application is shut down
        and the task is cancelled.

    Raises
    ------
    asyncio.CancelledError
        When the application shuts down and the broadcast task is cancelled.

    Notes
    -----
    - `args.hz` controls the pacing of the loop; it does not change the board
      sampling rate (which is fixed by hardware/board configuration).
    - `board.get_current_board_data(args.n)` returns the most recent samples
      currently available in the ring buffer, up to `n`.
    - If a WebSocket send fails, that client is considered dead and removed.

    Example
    -------
    Running the publisher using a synthetic board at 20 Hz, sending 32 samples
    per message on port 8080:

    >>> python eeg_acquisition.py --board 0 --hz 20 --n 32 --port 8080
    """
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
    """
    This request handler upgrades an HTTP request to a WebSocket connection and
    registers the client for EEG streaming.

    The handler is responsible for:
    - Creating and preparing an aiohttp WebSocketResponse.
    - Adding the WebSocket connection to the global `clients` registry.
    - Keeping the connection open by consuming incoming messages (if any).
    - Removing the client from the registry when the connection closes.

    Parameters
    ----------
    request : aiohttp.web.Request
        Incoming HTTP request to be upgraded to a WebSocket connection.

    Returns
    -------
    ws : aiohttp.web.WebSocketResponse
        The established WebSocket response object.

    Raises
    ------
    Exception
        Any unexpected aiohttp/WebSocket errors may propagate depending on
        server runtime conditions (network failures, client disconnects, etc.).

    Notes
    -----
    - Incoming client messages are ignored; the server acts as a publisher only.
    - Client removal is guaranteed via the `finally` block.

    Example
    -------
    A client can connect to:

    - ws://localhost:<port>/ws

    and receive periodic "eeg_chunk" JSON messages.
    """
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
    """
    This startup hook schedules the EEG broadcast coroutine as a background task
    when the aiohttp application starts.

    The hook is responsible for:
    - Creating an asyncio Task from `broadcast()`.
    - Storing the task inside the aiohttp application state under `app["task"]`.

    Parameters
    ----------
    app : aiohttp.web.Application
        The aiohttp application instance being started.

    Returns
    -------
    None

    Notes
    -----
    - The created task runs until cancelled during application cleanup.
    - The task reference is stored so it can be cancelled reliably on shutdown.

    Example
    -------
    The hook is registered via:

    >>> app.on_startup.append(on_startup)
    """
    app["task"] = asyncio.create_task(broadcast())


async def on_cleanup(app: web.Application) -> None:
    """
    This cleanup hook stops the EEG streaming task and releases BrainFlow
    resources when the aiohttp application shuts down.

    The hook is responsible for:
    - Cancelling the broadcast task created at startup.
    - Attempting to stop the BrainFlow stream and release the session.
    - Suppressing shutdown exceptions to avoid blocking server teardown.

    Parameters
    ----------
    app : aiohttp.web.Application
        The aiohttp application instance being cleaned up.

    Returns
    -------
    None

    Raises
    ------
    Exception
        Exceptions from BrainFlow shutdown calls are caught and suppressed.

    Notes
    -----
    - Task cancellation is requested with `cancel()`. If you want stricter
      shutdown guarantees, you could also `await app["task"]` and handle
      `asyncio.CancelledError`.
    - BrainFlow teardown is wrapped in try/except to tolerate partial shutdown
      states (e.g., stream already stopped).

    Example
    -------
    The hook is registered via:

    >>> app.on_cleanup.append(on_cleanup)
    """
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
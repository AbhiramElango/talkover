"""Side-by-side live demo: stock Pipecat (left) vs Talkover (right).

    uv pip install -e '.[demo]'
    .venv/bin/python demo/server.py        # then open http://localhost:7860

Needs DEEPGRAM_API_KEY and GROQ_API_KEY in .env.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pipecat.transports.smallwebrtc.request_handler import SmallWebRTCRequest, SmallWebRTCRequestHandler

from bot import StockPolicy, TalkoverPolicy, run_bot
from talkover.interruption.paths import data_dir
from talkover.interruption.runtime import ModelBundle

ROOT = Path(__file__).resolve().parent


def create_app(bundle: ModelBundle) -> FastAPI:
    policies = {policy.name: policy for policy in (StockPolicy(), TalkoverPolicy(bundle))}
    handler = SmallWebRTCRequestHandler()
    app = FastAPI()

    @app.get("/api/config")
    async def config() -> dict:
        return {"threshold": bundle.threshold}

    @app.post("/api/offer")
    async def offer(body: dict, background: BackgroundTasks) -> dict:
        request = SmallWebRTCRequest.from_dict(body)
        policy = policies.get((request.request_data or {}).get("side"))
        if policy is None:
            raise HTTPException(400, "side must be 'stock' or 'talkover'")

        async def start(connection) -> None:
            background.add_task(run_bot, connection, policy)

        return await handler.handle_web_request(request, start)

    @app.middleware("http")
    async def no_cache(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"   # the page changes often; never run a stale app.js
        return response

    app.mount("/", StaticFiles(directory=ROOT / "static", html=True))
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=data_dir() / "release" / "talkover")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    load_dotenv(ROOT.parent / ".env")
    uvicorn.run(create_app(ModelBundle.load(args.bundle)), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()

"""M8: routing service. Contract: docs/contracts.md#m8. Run: python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

Local only: no LLM, no external calls. Request text is customer data: it is never logged, stored or put in an error.
"""
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, field_validator
from pydantic_core import PydanticCustomError

from src import config
from src.predictions import load_final
from src.router import Router

INDEX_HTML = Path(__file__).parent / "static" / "index.html"
PLACEHOLDER = "__OPTIONS__"
OPTIONS = {
    "channel": config.CHANNELS,
    "product_family": config.PRODUCT_FAMILIES,
    "warranty_status": config.WARRANTY_STATUSES,
}
NO_STORE = {"Cache-Control": "no-store"}  # reasons quote the request
UNKNOWN_PRODUCT = "product not recognised; routed without the product"
NOT_READY = (
    "Sorry, the routing model is not available yet. Please run the train step (see \"Run the service\" in the README) and "
    "restart the service."
)
FAILED = "Sorry, this request could not be routed because of an internal error. Please try again."
BAD_BODY = "The request body must be a JSON object with a request_text field."
MESSAGES = {  # (field, error type) -> message; nothing from the request is ever added
    ("request_text", "missing"): "request_text is required.",
    ("request_text", "blank"): "request_text must not be empty.",
    ("request_text", "too_long"): f"request_text must be at most {config.MAX_REQUEST_CHARS:,} characters.",
    ("request_text", "unreadable"): "request_text contains a character that cannot be read.",
    ("request_text", "string_type"): "request_text must be text.",
    ("product_family", "string_type"): "product_family must be text, or left out.",
    **{
        (name, kind): f"{name} must be one of: {', '.join(OPTIONS[name])}; or left out."
        for name in ("channel", "warranty_status")
        for kind in ("unknown", "string_type")
    },
}
FIELDS = ("request_text", *OPTIONS)

log = logging.getLogger("kestrel.service")


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, hide_input_in_errors=True)

    request_text: str
    channel: str | None = None
    product_family: str | None = None
    warranty_status: str | None = None

    @field_validator("request_text")
    @classmethod
    def _text(cls, value: str) -> str:
        if len(value) > config.MAX_REQUEST_CHARS:
            raise PydanticCustomError("too_long", "too long")
        if not value.strip():
            raise PydanticCustomError("blank", "blank")
        if any(0xD800 <= ord(char) <= 0xDFFF for char in value):  # half a character: valid JSON, not valid text
            raise PydanticCustomError("unreadable", "unreadable")
        return value

    @field_validator("channel", "warranty_status")
    @classmethod
    def _known(cls, value: str | None, info) -> str | None:
        value = (value or "").strip() or None
        if value is not None and value not in OPTIONS[info.field_name]:
            raise PydanticCustomError("unknown", "unknown value")
        return value

    @field_validator("product_family")
    @classmethod
    def _product(cls, value: str | None) -> str | None:
        return (value or "").strip() or None


def _load(models_dir: Path) -> Router | None:
    """The final model, or None if it cannot be used for any reason: the service must start anyway."""
    try:
        return load_final(models_dir)
    except Exception as error:
        log.error("model not loaded (%s): run the train step", type(error).__name__)
        return None


def options_json(options: dict) -> str:
    """JSON that is safe inside a <script> element."""
    return json.dumps(options).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Fixed messages only: FastAPI's default answer repeats the submitted values."""
    messages = []
    for error in exc.errors():
        loc = error["loc"]
        field = loc[1] if len(loc) > 1 and loc[1] in FIELDS else None
        message = MESSAGES.get((field, error["type"])) if field else BAD_BODY
        messages.append(message or f"{field} is not valid text.")
    return JSONResponse({"detail": " ".join(dict.fromkeys(messages))}, status_code=422, headers=NO_STORE)


def create_app(models_dir: Path = config.MODELS_DIR) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.router = _load(models_dir)
        yield

    # The interactive docs pages load scripts from a CDN.
    app = FastAPI(title="Kestrel Home request router", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.router = None
    app.add_exception_handler(RequestValidationError, _validation_error)
    page_html = INDEX_HTML.read_text(encoding="utf-8").replace(PLACEHOLDER, options_json(OPTIONS))

    @app.get("/health")
    def health() -> dict:
        router = app.state.router
        window = router.meta["training_window"] if router else None
        return {
            "status": "ok",
            "model_loaded": router is not None,
            "model_version": router.meta["model_version"] if router else None,
            "training_window": f"{window['first_day']} to {window['last_day']}" if window else None,
        }

    @app.post("/predict")
    def predict(body: PredictRequest) -> JSONResponse:
        router = app.state.router
        if router is None:
            raise HTTPException(503, NOT_READY, headers=NO_STORE)
        try:
            output = router.predict([body.model_dump(exclude_none=True)])[0]
        except Exception as error:
            # The class only: messages and tracebacks can hold request text.
            log.error("prediction failed (%s)", type(error).__name__)
            return JSONResponse({"detail": FAILED}, status_code=500, headers=NO_STORE)
        if body.product_family is not None and body.product_family not in config.PRODUCT_FAMILIES:
            output = {**output, "reasons": [*output["reasons"], UNKNOWN_PRODUCT]}
        return JSONResponse(output, headers=NO_STORE)

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return page_html

    return app


app = create_app()

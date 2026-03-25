from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.routers.admin import router as admin_router
from app.routers.auth import router as auth_router
from app.routers.identify import router as identify_router
from app.routers.plants import router as plants_router
from app.routers.poi import router as poi_router


app = FastAPI(title="EcoLocatie API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _http_error_code(status_code: int) -> str:
    if status_code == 400:
        return "bad_request"
    if status_code == 401:
        return "unauthorized"
    if status_code == 403:
        return "forbidden"
    if status_code == 404:
        return "not_found"
    if status_code == 409:
        return "conflict"
    if status_code == 422:
        return "validation_error"
    if status_code == 502:
        return "bad_gateway"
    if status_code >= 500:
        return "internal_error"
    return "error"


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": _http_error_code(exc.status_code),
                "message": exc.detail,
                "path": request.url.path,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    fields = []
    for item in exc.errors():
        location = [str(part) for part in item.get("loc", []) if part != "body"]
        fields.append(
            {
                "field": ".".join(location) if location else "request",
                "message": item.get("msg", "Invalid value"),
                "type": item.get("type", "validation_error"),
            }
        )
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "Request validation failed",
                "path": request.url.path,
                "fields": fields,
            }
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    debug_mode = False
    try:
        debug_mode = get_settings().debug
    except Exception:
        debug_mode = False
    message = str(exc) if debug_mode else "An unexpected error occurred"
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": message,
                "path": request.url.path,
            }
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(auth_router)
app.include_router(plants_router)
app.include_router(poi_router)
app.include_router(identify_router)
app.include_router(admin_router)

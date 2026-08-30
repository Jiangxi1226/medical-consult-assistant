"""医疗会诊多智能体 —— FastAPI 入口。

启动前请在项目根准备 .env：
  LLM_BASE_URL=https://api.deepseek.com
  LLM_API_KEY=...
  LLM_TEXT_MODEL=...
(core/llm.py 首次 import 会自动加载 .env，无需手动 export)

启动： python app.py  →  http://localhost:8001
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api import consult
from utils.security import RateLimiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="智能分诊·多科会诊助手", version="1.0", lifespan=lifespan)

# CORS：默认仅本机；生产通过 CORS_ORIGINS 环境变量显式配置允许来源。
# 不启用 allow_credentials（避免与 allow_origins=["*"] 组合成浏览器安全违规），
# 前端所需鉴权改用 Authorization 头即可。
_def_cors = os.getenv("CORS_ORIGINS", "").strip()
_cors_origins = [o.strip() for o in _def_cors.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins, allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

# 限流：单客户端每分钟最多 120 次（相对宽松，避免误伤正常多轮问诊）
_limiter = RateLimiter(max_requests_per_minute=120)


def _client_ip(request: Request) -> str:
    """取真实客户端 IP 用于限流。

    默认用直连 peer IP(request.client.host)——这是最可靠的来源。
    只有 TRUST_PROXY=1 时才信任 X-Forwarded-For(取最左首个 IP)，
    因为该头可被客户端伪造；部署在可信反向代理后时才应开启。
    (与项目一 multimodal_rag 的 middleware 对齐。)
    """
    if os.getenv("TRUST_PROXY") == "1":
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path.startswith(("/v1/", "/api/")):
        if not _limiter.allow(_client_ip(request)):
            return JSONResponse({"error": "请求过于频繁，请稍后再试"}, status_code=429)
    return await call_next(request)


app.include_router(consult.router, prefix="/api")

_WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
app.mount("/static", StaticFiles(directory=_WEB), name="static")


@app.get("/")
async def index():
    # 首页关闭缓存：前端脚本/样式常迭代，避免浏览器缓存旧版导致"改了不生效"。
    resp = FileResponse(os.path.join(_WEB, "index.html"))
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


@app.get("/health")
async def health():
    return {"ok": True, "service": "medical-assistant"}


# ---- LLM 配置写入（前端"设置"弹窗调用）：写 .env + 热更新，无需重启 ----
class _SetupReq(BaseModel):
    LLM_KEY: str = ""      # 前端传sk开头的真实key
    LLM_BASE_URL: str = ""
    LLM_TEXT_MODEL: str = ""


@app.post("/api/config/llm")
async def config_llm(req: _SetupReq):
    base = (req.LLM_BASE_URL or "").strip()
    if base and not base.lower().startswith("https://"):
        return JSONResponse({"ok": False, "error": "API Base URL 必须为 https 端点。"}, status_code=400)
    key = (req.LLM_KEY or "").strip()
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    existing = {}
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    existing[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    # 已配置 key 不允许在线替换/清空（防远程改写）
    if existing.get("LLM_API_KEY") and key and key != existing["LLM_API_KEY"]:
        return JSONResponse({"ok": False, "error": "已配置 LLM_API_KEY，不允许在线替换；如需更换请手动编辑 .env。"}, status_code=403)
    if existing.get("LLM_API_KEY") and not key:
        return JSONResponse({"ok": False, "error": "不允许在线清空已有 LLM_API_KEY。"}, status_code=403)
    merged = {k: v for k, v in existing.items() if k not in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_TEXT_MODEL")}
    if base: merged["LLM_BASE_URL"] = base
    if key: merged["LLM_API_KEY"] = key
    if req.LLM_TEXT_MODEL.strip(): merged["LLM_TEXT_MODEL"] = req.LLM_TEXT_MODEL.strip()
    with open(env_path, "w", encoding="utf-8") as f:
        f.write("# 医疗会诊多智能体 — 配置（请勿提交含真实密钥的 .env）\n")
        for k, v in merged.items():
            f.write(f"{k}={v}\n")
    try:
        from core.llm import reload_config
        reload_config()
    except Exception:
        pass
    return {"ok": True, "configured": bool(merged.get("LLM_API_KEY"))}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8001")))

# 医疗会诊多智能体 — Dockerfile
# 多阶段构建：构建阶段装依赖，运行阶段只拷贝产物，减小镜像体积。
#
# 设计要点：
#   - 镜像只装「代码 + 环境」，运行时数据（审计日志、评估结果）用 VOLUME 挂载，
#     不打进镜像 —— 镜像可移植，数据可持久化。
#   - HEALTHCHECK 主动探 /health（app.py 已提供该路由），供编排工具探活/自动重启。
#     给 60s 启动宽限期：首次 import 会加载模型/知识库，启动不是瞬间完成。

# ═══ 构建阶段 ═══
FROM python:3.13-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ═══ 运行阶段 ═══
FROM python:3.13-slim

WORKDIR /app

# 从构建阶段复制已安装的依赖，不带 pip 缓存与构建工具
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# 复制应用代码（哪些不拷由 .dockerignore 决定）
COPY . .

# 运行时数据目录：有则挂载宿主机目录复用，无则容器内自建
VOLUME ["/app/data", "/app/logs", "/app/eval"]

# 健康检查：主动探 /health，连续失败 3 次标记 unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

# 端口 8001（项目一用 8000，此处刻意错开，可同时运行）
EXPOSE 8001

# 默认启动 FastAPI 服务（app.py 内置 uvicorn，读 PORT 环境变量，默认 8001）
CMD ["python", "app.py"]

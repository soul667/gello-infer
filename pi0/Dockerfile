FROM python:3.7-slim
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

# 安装所有系统依赖
RUN apt-get update && apt-get install -y \
    git \
    linux-libc-dev \
    build-essential \
    clang \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 设置环境变量
ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/.venv \
    VIRTUAL_ENV=/.venv \
    PATH="/.venv/bin:$PATH" \
    UV_HTTP_TIMEOUT=300

# 复制所有需要的文件
COPY ./openpi/examples/simple_client/requirements.txt /tmp/requirements.txt
COPY ./openpi/packages/openpi-client/pyproject.toml /tmp/openpi-client/pyproject.toml
COPY ./gello_software/requirements.txt /tmp/gello-requirements.txt
COPY ./gello_software /opt/gello_software
COPY ./openpi /tmp/openpi

# 创建虚拟环境
RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT

# 安装基础依赖（分层以便缓存）
RUN uv pip install -r /tmp/gello-requirements.txt && \
    uv pip install opencv-python-headless

# 安装 gello 及其依赖
RUN cd /opt/gello_software/third_party/DynamixelSDK/python && uv pip install . && \
    cd /opt/gello_software && uv pip install -e .

# 安装 openpi 依赖
RUN uv pip install -r /tmp/requirements.txt && \
    cd /tmp/openpi/packages/openpi-client && uv pip install -e . && \
    cd /tmp/openpi && uv pip install -e .

# 清理临时文件
RUN rm -f /tmp/gello-requirements.txt /tmp/requirements.txt



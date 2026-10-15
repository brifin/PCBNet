# 赛题六提交镜像构建文件（按 2606sample_submission 的 Dockerfile 约定）
# 平台会 docker build 该文件 → ENTRYPOINT ["/submission/run.sh"] → run.sh -input /input -output /output
# 基础镜像以赛题页"资源下载"为准；样本实例用 edarace-agent-base，可用 --build-arg 覆盖
ARG BASE_IMAGE=edarace-agent-base:ubuntu24
FROM ${BASE_IMAGE}

WORKDIR /submission

# 复制所有代码 / 权重 / 知识库 / 预置 wheels
COPY . /submission/

# 安装依赖（离线优先：用 /submission/wheels 里的 wheel；缺的才联网，带长超时重试）
# 基础镜像已装的包（如 torch）不会被重装，避免构建时访问 pypi 超时
RUN python3 /submission/install_deps.py

# 预置 OCR 模型（运行时容器无外网，paddleocr 不能现场下载；模型已随 zip 打包）
COPY ocr_models/.paddleocr /root/.paddleocr

RUN chmod +x /submission/run.sh

# OCR 固定用 CPU（评测机是 A800，paddle CPU 识别足够；不占 GPU 显存、无 cudnn 冲突）
ENV PCBOCR_DEVICE=cpu

ENTRYPOINT ["/submission/run.sh"]

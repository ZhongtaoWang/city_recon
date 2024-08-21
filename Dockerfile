
FROM nvidia/cuda:11.3.1-devel-ubuntu18.04 


RUN chmod 1777 /tmp

RUN apt-key adv --fetch-keys http://archive.ubuntu.com/ubuntu/project/ubuntu-archive-keyring.gpg

WORKDIR /app


RUN apt-get update --allow-releaseinfo-change && apt-get remove --purge python3 && apt-get autoremove

# 更新源并安装基础依赖
RUN apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-distutils \
    curl

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.8 get-pip.py

# 设置 Python3 的别名
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 

# 安装 PyTorch及其依赖
RUN  pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --index-url https://download.pytorch.org/whl/cu113

RUN pip install opencv-python huggingface_hub matplotlib pycocotools opencv-python onnx onnxruntime timm

# 安装其他 Python 依赖
RUN pip install -U openmim && \
    mim install  mmcv-full && \
    pip install mmsegmentation==0.30.0 && \
    pip install scikit-image scikit-learn scipy

# 复制当前目录内容到容器中
COPY . .

# 安装第三方工具的依赖
RUN cd third_party/sam-hq && \
    pip install . && \
    cd ../../third_party/AerialFormer && \
    pip install . && cd ../../

RUN cp myrun.py third_party/Depth-Anything

#fix broken runtime
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0

# 复制预训练模型权重
#COPY weights/ /app/weights/

# 将入口点设置为启动脚本 recon.sh
ENTRYPOINT ["sh", "recon.sh"]

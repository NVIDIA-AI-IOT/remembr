FROM nvcr.io/nvstaging/isaac-amr/vila

WORKDIR /app

RUN curl -fsSL https://ollama.com/install.sh | sh
RUN pip install langchain-community langgraph langchain_openai langchain_nvidia_ai_endpoints
RUN pip install pymilvus gradio==4.36.1

RUN conda init bash
RUN source ~/.bashrc
RUN export PATH="/app/VILA:$PATH"

RUN echo "source activate vila" > ~/.bashrc
ENV PATH /opt/conda/envs/{env}/bin:$PATH


# Install ROS
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install ros-humble-desktop -y

# this broke things?
RUN conda uninstall cuda-toolkit
# RUN conda install -c nvidia cuda-toolkit
RUN  conda install -c conda-forge gcc=12.1.0

COPY ./data long_horizon_perception/data

COPY ./ long_horizon_perception/

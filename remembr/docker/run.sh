docker pull nvcr.io/nvstaging/isaac-amr/long_horizon_perception
docker run --gpus all --network=host -it \
    -e LLM_GATEWAY_CLIENT_ID \
    -e LLM_GATEWAY_CLIENT_SECRET \
    -v .:/app/long_horizon_perception \
    nvcr.io/nvstaging/isaac-amr/long_horizon_perception \
    # cd long_horizon_perception; \
    # python chat_demo/demo.py --chatbot_host_ip localhost --llm_backend gpt-4o
    /bin/bash
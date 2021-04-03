FROM ubuntu:20.10
RUN apt-get update && apt-get install nvidia-cuda-toolkit git build-essential wget -y

## Compile offloaded gcc
RUN mkdir -p /offload
COPY compile.sh /offload 
WORKDIR /offload
RUN chmod +x compile.sh
RUN ./compile.sh

## Copy App for test (add a volume later)
WORKDIR /app
COPY . .

ENTRYPOINT ["/bin/bash"]

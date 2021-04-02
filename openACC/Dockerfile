FROM ubuntu:18.04
RUN apt-get update && apt-get install git build-essential wget software-properties-common -y

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

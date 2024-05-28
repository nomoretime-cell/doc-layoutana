#!/bin/bash

latest_tag=$(git describe --tags --abbrev=0)
module_name=doc-layoutana

docker run \
  -d -it \
  --gpus device=0 \
  -e WORKER_NUM=2 \
  -p 8001:8001 \
  --name ${module_name} \
  ${module_name}:${latest_tag}

#!/bin/bash

latest_tag=$(git describe --tags --abbrev=0)
module_name=doc-layoutana

docker build \
  -t ${module_name}:${latest_tag} \
  -f Dockerfile ..

docker save -o images/${module_name}_${latest_tag}.tar ${module_name}:${latest_tag}
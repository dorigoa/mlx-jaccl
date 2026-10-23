#!/bin/bash

mlx_vlm.server \
  --model mlx-community/Muse-Glimmer-30B-4bit \
  --enable-thinking \
  --max-tokens 262144 \
  --port 8080 --host 0.0.0.0 
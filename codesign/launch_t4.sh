#!/bin/bash
# Spin a SPOT T4 from the baked custom image `infinity-2b-t4` — NO download, NO pip.
# Weights (2b+vae+flan-t5-xl), deps, and the repo are already in the image (~/Infinity,
# ~/.cache/huggingface). SPOT + self-delete safety. Usage: ./launch_t4.sh [name]
set -uo pipefail
PROJECT=project-e7987ca9-ebd3-438f-95f
NAME=${1:-infinity-t4-bench}
for ZONE in us-central1-a us-central1-b us-east1-c us-east1-d us-west1-a us-west1-b; do
  echo ">>> $ZONE ..."
  if gcloud compute instances create "$NAME" --project=$PROJECT --zone=$ZONE \
      --machine-type=n1-standard-8 --accelerator=type=nvidia-tesla-t4,count=1 \
      --provisioning-model=SPOT --instance-termination-action=DELETE --max-run-duration=3h \
      --image=infinity-2b-t4 --image-project=$PROJECT \
      --boot-disk-size=100GB --boot-disk-type=pd-balanced >/tmp/launch.log 2>&1; then
    echo "UP: $NAME in $ZONE"
    echo "next: gcloud compute ssh $NAME --zone=$ZONE --project=$PROJECT"
    echo "then: cd ~/Infinity && git pull && <run sweep>"
    exit 0
  else
    echo "  failed: $(tail -1 /tmp/launch.log)"
  fi
done
echo "all zones failed (capacity/quota)"; exit 1

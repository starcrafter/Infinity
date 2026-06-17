#!/bin/bash
# Delete the SPOT T4 (boot disk auto-deletes). ALWAYS run when done — quota is 1 GPU,
# shared with the faro project. Usage: ./teardown_t4.sh [name] [zone]
set -uo pipefail
PROJECT=project-e7987ca9-ebd3-438f-95f
NAME=${1:-infinity-t4-bench}
ZONE=${2:-}
[ -z "$ZONE" ] && ZONE=$(gcloud compute instances list --project=$PROJECT \
    --filter="name=$NAME" --format="value(zone)" 2>/dev/null | sed 's#.*/##')
[ -z "$ZONE" ] && { echo "instance $NAME not found (already gone?)"; exit 0; }
gcloud compute instances delete "$NAME" --zone="$ZONE" --project=$PROJECT -q
echo "deleted $NAME; GPU quota usage:"
gcloud compute project-info describe --project $PROJECT --format="json(quotas)" 2>/dev/null | \
  python3 -c "import json,sys; d=json.load(sys.stdin); print([q for q in d['quotas'] if q['metric']=='GPUS_ALL_REGIONS'][0])"

#!/bin/bash
end_time=$((SECONDS + 14400))
while [ $SECONDS -lt $end_time ]; do
    echo -n .
    sleep 120
done
echo "Keepalive stopped after 4 hours"

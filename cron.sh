#!/bin/sh
# * * * * * python /mnt/d/workspace/oraculus/get_data.py >> /tmp/get_data.log 
echo "Cron."
while true; do
    python /mnt/d/workspace/oraculus/get_data.py >> /tmp/get_data.log
    sleep 60
done

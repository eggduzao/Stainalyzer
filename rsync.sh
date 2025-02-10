#!/bin/bash

# Define variables
LOCAL_PATH="/Users/egg/Projects/Stainalyzer/data"
REMOTE_USER="egusmao"
REMOTE_HOST="192.168.25.2"
REMOTE_PATH="/storage2/egusmao/projects/Stainalyzer"
DIRECTION=$1  # 'send' or 'receive'

# Rsync options
RSYNC_OPTIONS="-avz --progress"

# Check direction
if [ "$DIRECTION" == "send" ]; then
    echo "Sending files to the cluster..."
    rsync $RSYNC_OPTIONS "$LOCAL_PATH" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
elif [ "$DIRECTION" == "receive" ]; then
    echo "Receiving files from the cluster..."
    rsync $RSYNC_OPTIONS "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH" "$LOCAL_PATH"
else
    echo "Usage: $0 {send|receive}"
    exit 1
fi

echo "Rsync operation completed."

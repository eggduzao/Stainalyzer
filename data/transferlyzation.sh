#!/bin/bash

# Set your variables here
LOCAL_USER="egg"  # Your local machine username
LOCAL_ADDRESS="192.168.0.3"  # Your local machine IP address
CLUSTER_PATH="/storage2/egusmao/projects/Stainalyzer/data/"
HOME_PATH="/Users/egg/Projects/Stainalyzer/data/"

# Data file name
DATA_FILE="DAB_IMIP_Tratamento_Clean.tar.gz"

# Function to pull data from home to the cluster (resumable)
to_cluster() {
    echo "Pulling data from HOME to CLUSTER (resumable)..."
    rsync -avP "$LOCAL_USER@$LOCAL_ADDRESS:$HOME_PATH/$DATA_FILE" "$CLUSTER_PATH"
    
    if [ $? -eq 0 ]; then
        echo "Transfer to cluster completed successfully!"
    else
        echo "Error during transfer to cluster. You can re-run the script to resume."
    fi
}

# Function to push data from the cluster to home (resumable)
to_home() {
    echo "Pushing data from CLUSTER to HOME (resumable)..."
    rsync -avP "$CLUSTER_PATH/$DATA_FILE" "$LOCAL_USER@$LOCAL_ADDRESS:$HOME_PATH"
    
    if [ $? -eq 0 ]; then
        echo "Transfer to home completed successfully!"
    else
        echo "Error during transfer to home. You can re-run the script to resume."
    fi
}

# Check for user input
if [ "$1" == "to_cluster" ]; then
    to_cluster
elif [ "$1" == "to_home" ]; then
    to_home
else
    echo "Usage: $0 [to_cluster | to_home]"
    exit 1
fi

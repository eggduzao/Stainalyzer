#!/bin/bash

# Set your variables here
CLUSTER_USER="egusmao"
CLUSTER_ADDRESS="192.168.25.2"
CLUSTER_PATH="/storage2/egusmao/projects/Stainalyzer/data/"
HOME_PATH="/Users/egg/Projects/Stainalyzer/data/"

# Data file name
DATA_FILE="DAB_IMIP_Tratamento_Clean.tar.gz"

# Function to transfer data to the cluster
to_cluster() {
    echo "Transferring data from HOME to CLUSTER..."
    rsync -avz "$HOME_PATH/$DATA_FILE" "$CLUSTER_USER@$CLUSTER_ADDRESS:$CLUSTER_PATH"
    
    if [ $? -eq 0 ]; then
        echo "Transfer to cluster completed successfully!"
    else
        echo "Error during transfer to cluster."
    fi
}

# Function to transfer data to home
to_home() {
    echo "Transferring data from CLUSTER to HOME..."
    rsync -avz "$CLUSTER_USER@$CLUSTER_ADDRESS:$CLUSTER_PATH/$DATA_FILE" "$HOME_PATH"
    
    if [ $? -eq 0 ]; then
        echo "Transfer to home completed successfully!"
    else
        echo "Error during transfer to home."
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

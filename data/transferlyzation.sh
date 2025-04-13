#!/bin/bash

# === CONFIGURATION ===
CLUSTER_USER="egusmao"
CLUSTER_ADDRESS="192.168.25.2"
CLUSTER_PATH="/storage2/egusmao/projects/Stainalyzer/data/results/Neila_DAB/"

HOME_USER="egg"
HOME_PATH="/Users/egg/projects/Stainalyzer/data/results/Neila_DAB/"

DATA_FILE="DAB_IMIP_Tratamento"
SSH_KEY="$HOME/.ssh/id_ed25519"

# === DETECTION LOGIC ===
IS_CLUSTER=false
hostname | grep -q "carloschagas" && IS_CLUSTER=true

# === FUNCTIONS ===

to_cluster() {
    echo "Transferring from HOME → CLUSTER..."
    rsync -avP "$HOME_PATH/$DATA_FILE" "$CLUSTER_USER@$CLUSTER_ADDRESS:$CLUSTER_PATH"
    [ $? -eq 0 ] && echo "Transfer complete!" || echo "Error: Transfer failed."
}

to_home() {
    echo "Transferring from CLUSTER → HOME..."
    rsync -avP "$CLUSTER_USER@$CLUSTER_ADDRESS:$CLUSTER_PATH/$DATA_FILE" "$HOME_PATH"
    [ $? -eq 0 ] && echo "Transfer complete!" || echo "Error: Transfer failed."
}

# === EXECUTION LOGIC ===

if [ "$IS_CLUSTER" = true ]; then
    echo "Error: This script must be run from your local machine (not from the cluster)."
    exit 1
fi

if [ "$1" == "to_cluster" ]; then
    to_cluster
elif [ "$1" == "to_home" ]; then
    to_home
else
    echo "Usage: $0 [to_cluster | to_home]"
    exit 1
fi


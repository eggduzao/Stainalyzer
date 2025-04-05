#!/bin/bash

# === CONFIGURATION ===
CLUSTER_USER="egusmao"
CLUSTER_ADDRESS="192.168.25.2"
CLUSTER_PATH="/storage2/egusmao/projects/Stainalyzer/data/results/"

HOME_USER="egg"
HOME_ADDRESS="192.168.15.4"
# HOME_ADDRESS=$(ipconfig getifaddr en0)
HOME_PATH="/Users/egg/cluster_projects/projects/Stainalyzer/data/results/"

DATA_FILE="Neila_DAB"
SSH_KEY_CLUSTER_TO_HOME="$HOME/.ssh/id_ed25519_cluster_to_home"

# === DETECTION LOGIC ===
IS_CLUSTER=false
IS_HOME=false
hostname | grep -q "carloschagas" && IS_CLUSTER=true
hostname | grep -q "MacBookPro" && IS_HOME=true 

# === FUNCTIONS ===

to_cluster() {
    echo "Transferring from HOME → CLUSTER..."
    rsync -avP "$HOME_PATH/$DATA_FILE" "$CLUSTER_USER@$CLUSTER_ADDRESS:$CLUSTER_PATH"
    [ $? -eq 0 ] && echo "Transfer complete!" || echo "Error: Transfer failed."
}

to_home() {
    echo "Transferring from CLUSTER → HOME..."
    rsync -e "ssh -i $SSH_KEY_CLUSTER_TO_HOME" -avP "$CLUSTER_PATH/$DATA_FILE" "$HOME_USER@$HOME_ADDRESS:$HOME_PATH"
    [ $? -eq 0 ] && echo "Transfer complete!" || echo "Error: Transfer failed."
}

# === EXECUTION LOGIC ===

if [ "$1" == "to_cluster" ]; then
    if [ "$IS_CLUSTER" = true ]; then
        echo "Warning: You are already on the cluster. Cannot send data TO the cluster from here."
        exit 1
    else
        to_cluster
    fi
elif [ "$1" == "to_home" ]; then
    if [ "$IS_HOME" = true ]; then
        to_home
    else
        echo "Warning: You are on your local machine. Cannot PULL data from the cluster using this mode."
        echo "Use 'to_cluster' to push instead."
        exit 1
    fi
else
    echo "Usage: $0 [to_cluster | to_home]"
    exit 1
fi


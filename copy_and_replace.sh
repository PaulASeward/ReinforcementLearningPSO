#!/bin/bash

# Get the list of numbers from the first argument, split by comma
IFS=',' read -ra NUMBERS <<< "$1"

# Get the current date in YYYYMMDD format
DATE=$(date +%Y%m%d)

# Loop over each number
for i in "${NUMBERS[@]}"; do
    # Create a new directory with the date, 'f', and the number as part of the name
    NEW_DIR="run_history/${DATE}_f$i"
    mkdir -p "$NEW_DIR"

    # Copy the directories and files into the new directory
    rsync -av --exclude='__pycache__' agents "$NEW_DIR"
    rsync -av --exclude='__pycache__' environment "$NEW_DIR"
    rsync -av --exclude='__pycache__' pso "$NEW_DIR"
    rsync -av --exclude='__pycache__' utils "$NEW_DIR"
    cp main.py "$NEW_DIR"
    cp config.py "$NEW_DIR"

    # Copy the run_agent.sh file and replace '12' with the current number
    sed "s/--func_num=12/--func_num=$i/g; s/--job-name=f12/--job-name=f$i/g" run_agent.sh > "$NEW_DIR/run_agent.sh"

    # Make the new run_agent.sh file executable
    chmod +x "$NEW_DIR/run_agent.sh"
done
#!/bin/bash

base=$1
dataname=$2
model=$3

echo "Starting judge.py with model: $model"

while true; do
    # Temporary file storage error output
    err_file=$(mktemp)
    python judge.py -n $dataname -m $model -b $base 2> "$err_file"
    exit_code=$?
    
    # Display error output (if any)
    if [ -s "$err_file" ]; then
        cat "$err_file" >&2
    fi
    
    # Check if the balance is insufficient
    if grep -q "NOT_ENOUGH_BALANCE" "$err_file"; then
        echo "❌ Insufficient balance (NOT_ENOUGH_BALANCE). Stopping."
        rm -f "$err_file"
        break
    fi
    
    rm -f "$err_file"
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ judge.py completed successfully."
        break
    fi
done
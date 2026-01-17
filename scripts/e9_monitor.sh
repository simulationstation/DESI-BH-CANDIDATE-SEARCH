#!/bin/bash
while true; do
    echo "$(date): Checking..."
    n_files=$(find /home/primary/MSOUP/data/e9_healpix -name "*.fits" -size +100M 2>/dev/null | wc -l)
    echo "  Complete files: $n_files"
    
    python3 scripts/run_e9_local_healpix.py 2>&1 | tail -10
    
    status=$(globus task show 8b63c462-f3d2-11f0-8d1c-02edfb1d9cf1 2>&1 | grep "Status:")
    echo "  Transfer $status"
    
    if [[ "$status" == *"SUCCEEDED"* ]]; then
        echo "Transfer complete!"
        break
    fi
    
    sleep 120
done

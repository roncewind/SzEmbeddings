#!/bin/bash
# Quick redoer status check

echo "Redoer Process Status:"
echo "======================"
ps aux | grep sz_simple_redoer | grep -v grep | awk '{print "PID: " $2 "  CPU: " $3 "%  Memory: " $4 "%  Time: " $10}'

echo ""
echo "Latest Stats (if available):"
echo "============================"
if [ -f redoer.log ]; then
    tail -1 redoer.log | python3 -m json.tool 2>/dev/null | grep -E "repairedEntities|reevaluations|addedRecords|datetimestamp" || echo "No JSON stats yet"
else
    echo "No log file yet"
fi

echo ""
echo "Database Status:"
echo "================"
PGPASSWORD=senzing psql -h localhost -U senzing -d senzing -t -c "
SELECT
    'Records: ' || COUNT(*) || ', Entities: ' || COUNT(DISTINCT RES_ENT_ID)
FROM OBS_ENT;
" 2>/dev/null || echo "Could not connect to database"

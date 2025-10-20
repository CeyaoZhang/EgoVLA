python -u -W ignore server.py --port 8000 \
    --model-path Efficient-Large-Model/VILA1.5-3B \
    --conv-mode vicuna_v1 2>&1 | tee server.log &

# Get the server process id by grepping the process name

server_pid=$(ps aux | grep "python -u -W ignore server.py" | grep -v grep | awk '{print $2}')
echo "Server process id: $server_pid"

sleep 5

if [[ -f server.log ]]; then
    # Listen to the last line of the log file contains the expected message
    # "Uvicorn running on"
    tail -f server.log | while read line; do
        if [[ $line == *"Uvicorn running on"* ]]; then
            echo "Server is up and running"
            break
        fi
    done
else
    echo "Server log file not found"
    exit 1
fi


python tests/utils/client.py

# Kill the server process
kill -9 $server_pid

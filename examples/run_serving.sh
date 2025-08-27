VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_LOGGING_LEVEL=DEBUG VLLM_NIXL_SIDE_CHANNEL_HOST=localhost VLLM_NIXL_SIDE_CHANNEL_PORT=15577 vllm serve Qwen/Qwen3-0.6B --gpu-memory-utilization 0.7 --host localhost --disable-log-stats --port 8100 --max-model-len 8192 --seed 42 --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' 2>&1 | tee nixl/nixl_server1.log &
pid_server1=$(($!-1))

until [[ "$n" -ge 1000 ]] || [[ $ready == true ]]; do
    n=$((n+1))
    if grep -q "Started server process" nixl/nixl_server1.log; then
        break
    fi
    sleep 5s
done

curl http://localhost:8100/v1/completions -H "Content-Type: application/json" -d '{"model": "Qwen/Qwen3-0.6B", "prompt": "What is AI? ", "max_tokens": 20}'

echo "server1 is ${pid_server1}"

kill -9 ${pid_server1}
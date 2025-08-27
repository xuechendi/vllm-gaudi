VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_NIXL_SIDE_CHANNEL_HOST=localhost VLLM_NIXL_SIDE_CHANNEL_PORT=15578 vllm serve Qwen/Qwen3-0.6B --gpu-memory-utilization 0.3 --host localhost --disable-log-stats --port 8100 --max-model-len 8192 --seed 42 --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' 2>&1 | tee nixl/nixl_server1.log &
pid_server1=$(($!-1))

until [[ "$n" -ge 1000 ]] || [[ $ready == true ]]; do
    n=$((n+1))
    if grep -q "Started server process" nixl/nixl_server1.log; then
        break
    fi
    sleep 5s
done

VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_NIXL_SIDE_CHANNEL_HOST=localhost VLLM_NIXL_SIDE_CHANNEL_PORT=15678 vllm serve Qwen/Qwen3-0.6B --gpu-memory-utilization 0.3 --host localhost --disable-log-stats --port 8200 --max-model-len 8192 --seed 42 --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' 2>&1 | tee nixl/nixl_server2.log &
pid_server2=$(($!-1))

until [[ "$n" -ge 1000 ]] || [[ $ready == true ]]; do
    n=$((n+1))
    if grep -q "Started server process" nixl/nixl_server2.log; then
        break
    fi
    sleep 5s
done

sleep 3
python3 vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py --prefiller-host localhost --prefiller-port 8100   --decoder-host localhost --decoder-port 8200   --host=localhost --port 18192 2>&1 | tee nixl/nixl_proxy.log &
pid_proxy=$(($!-1))

until [[ "$n" -ge 1000 ]] || [[ $ready == true ]]; do
    n=$((n+1))
    if grep -q "Started server process" nixl/nixl_proxy.log; then
        break
    fi
    sleep 5s
done

sleep 3
curl http://localhost:18192/v1/completions -H "Content-Type: application/json" -d '{"model": "Qwen/Qwen3-0.6B", "prompt": "What is AI? ", "max_tokens": 20}'
#curl http://localhost:8100/v1/completions -H "Content-Type: application/json" -d '{"model": "Qwen/Qwen3-0.6B", "prompt": "What is AI? ", "max_tokens": 20}'

echo "server1 is ${pid_server1}"
echo "server2 is ${pid_server2}"
echo "proxy is ${pid_proxy}"

kill -9 ${pid_server1}
kill -9 ${pid_server2}
kill -9 ${pid_proxy}
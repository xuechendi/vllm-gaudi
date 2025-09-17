NIXL_BUFFER_DEVICE=${NIXL_BUFFER_DEVICE:-"cpu"}
#HABANA_LOGS=nixl_server1 LOG_LEVEL_ALL=3 \
HMEM_SYNAPSEAI=1 RANK=0 VLLM_LOGGING_LEVEL=DEBUG VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_NIXL_SIDE_CHANNEL_HOST=localhost VLLM_NIXL_SIDE_CHANNEL_PORT=15578 \
vllm serve Qwen/Qwen3-0.6B --gpu-memory-utilization 0.3 --host localhost --disable-log-stats --port 8100 --max-model-len 8192 --seed 42 --enforce-eager \
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both", "kv_buffer_device":"'"${NIXL_BUFFER_DEVICE}"'"}' 2>&1 | tee nixl/nixl_server1.log &
pid_server1=$(($!-1))

until [[ "$n" -ge 1000 ]] || [[ $ready == true ]]; do
    n=$((n+1))
    if grep -q "Started server process" nixl/nixl_server1.log; then
        break
    fi
    sleep 5s
done

#HABANA_LOGS=nixl_server2 LOG_LEVEL_ALL=1 \
HMEM_SYNAPSEAI=1 RANK=1 VLLM_LOGGING_LEVEL=DEBUG VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_NIXL_SIDE_CHANNEL_HOST=localhost VLLM_NIXL_SIDE_CHANNEL_PORT=15678 \
vllm serve Qwen/Qwen3-0.6B --gpu-memory-utilization 0.3 --host localhost --disable-log-stats --port 8200 --max-model-len 8192 --seed 42 --enforce-eager \
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both", "kv_buffer_device":"'"${NIXL_BUFFER_DEVICE}"'"}' 2>&1 | tee nixl/nixl_server2.log &
pid_server2=$(($!-1))

until [[ "$n" -ge 1000 ]] || [[ $ready == true ]]; do
    n=$((n+1))
    if grep -q "Started server process" nixl/nixl_server2.log; then
        break
    fi
    sleep 5s
done

sleep 3
python3 examples/nixl/toy_proxy_server.py --prefiller-host localhost --prefiller-port 8100   --decoder-host localhost --decoder-port 8200   --host=localhost --port 18192 2>&1 | tee nixl/nixl_proxy.log &
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
# sleep 3
# curl http://localhost:18192/v1/completions -H "Content-Type: application/json" -d '{"model": "Qwen/Qwen3-0.6B", "prompt": "What is AI? ", "max_tokens": 20}'
# curl http://localhost:18192/v1/completions -H "Content-Type: application/json" -d '{"model": "Qwen/Qwen3-0.6B", "prompt": "What is AI? ", "max_tokens": 20}'
# curl http://localhost:18192/v1/completions -H "Content-Type: application/json" -d '{"model": "Qwen/Qwen3-0.6B", "prompt": "What is AI? ", "max_tokens": 20}'
#curl http://localhost:8100/v1/completions -H "Content-Type: application/json" -d '{"model": "Qwen/Qwen3-0.6B", "prompt": "What is AI? ", "max_tokens": 20}'

sleep 3

echo "server1 is ${pid_server1}"
echo "server2 is ${pid_server2}"
echo "proxy is ${pid_proxy}"

lsof -i :15578 | grep LISTEN | awk '{print $2}' | xargs kill -9
lsof -i :15678 | grep LISTEN | awk '{print $2}' | xargs kill -9

kill -9 ${pid_server1}
kill -9 ${pid_server2}
kill -9 ${pid_proxy} 

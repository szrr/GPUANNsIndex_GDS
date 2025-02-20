#!/bin/bash

# 开启命令执行追踪
set -x

# 编译
cmake -B build .
cd build 
make -j

# 定义变量
INDEX_PATH="/mnt/Samsung980PRO2TB/szr/starling/bigann_100m_M256_R64_L100_B256/_disk.index"
QUERY_PATH="/home/ErHa/graph_sift1m/bigann_query.bvecs"
GRAPH_PATH="/data/szr/dataset/sift1b/graph/finalgraph/SIFT1B_11_21_degree64.bin"
GROUNDTRUTH_PATH="/data/szr/dataset/sift1b/gnd/idx_100M.ivecs"
# OUTPUT_DIR="/home/ErHa/szr/anns_bam_res/warmup1_hnsw"
OUTPUT_DIR="/home/ErHa/szr/anns_bam_res/rvq"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 定义测试参数
CANDIDATE_LIST=(16 32 64 128 256 512 1024)
# BEAM_WIDTH=(1 2)
BEAM_WIDTH=(2)
CACHE_SIZE=(5242880)  # 分别是5M, 1G, 8G, 26G  1073741824 8589934592 27917287424

# 遍历参数组合
for beam in "${BEAM_WIDTH[@]}"; do
    for cache in "${CACHE_SIZE[@]}"; do
        # 构造输出文件名
        CACHE_HUMAN_READABLE=""
        if [ "$cache" -eq 5242880 ]; then
            CACHE_HUMAN_READABLE="5M"
        elif [ "$cache" -eq 1073741824 ]; then
            CACHE_HUMAN_READABLE="1G"
        elif [ "$cache" -eq 8589934592 ]; then
            CACHE_HUMAN_READABLE="8G"
        elif [ "$cache" -eq 27917287424 ]; then
            CACHE_HUMAN_READABLE="26G"
        fi
        OUTPUT_FILE="$OUTPUT_DIR/beamwidth${beam}_cache${CACHE_HUMAN_READABLE}.log"

        # 初始化日志文件并记录开始时间和参数
        if [ ! -f "$OUTPUT_FILE" ]; then
            echo "Test started at $(date)" > "$OUTPUT_FILE"
            echo "Beam width: $beam, Cache size: $CACHE_HUMAN_READABLE" >> "$OUTPUT_FILE"
            echo "--------------------------------------" >> "$OUTPUT_FILE"
        fi

        for candidate in "${CANDIDATE_LIST[@]}"; do
            # 在日志文件中记录当前参数
            echo "Running candidate list size: $candidate" >> "$OUTPUT_FILE"

            # 执行命令并将输出追加到文件
            sudo ./query "$INDEX_PATH" "$QUERY_PATH" "$GRAPH_PATH" "$GROUNDTRUTH_PATH" \
                $candidate $beam 10 100000000 128 64 $cache 8 128 >> "$OUTPUT_FILE"

            # 分隔不同 candidate 的输出
            echo "--------------------------------------" >> "$OUTPUT_FILE"
        done
    done
done

# 提示脚本运行完成
echo "All tests completed. Results are saved in $OUTPUT_DIR."

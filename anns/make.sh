# cd ../
# rm -rf build
cmake -B build .
cd build 
make -j
sudo ./query /mnt/Samsung980PRO2TB/szr/starling/bigann_100m_M256_R64_L100_B256/_disk.index /home/ErHa/graph_sift1m/bigann_query.bvecs /data/szr/dataset/sift1b/graph/finalgraph/SIFT1B_11_21_degree64.bin /data/szr/dataset/sift1b/gnd/idx_100M.ivecs 16 1 10 100000000 128 64 20971520 2 64 10000 4096
# sudo ./query /mnt/Samsung980PRO2TB/szr/starling/bigann_100m_M256_R64_L100_B256/_disk.index /home/ErHa/graph_sift1m/bigann_query.bvecs /data/szr/dataset/sift1b/graph/finalgraph/SIFT1B_11_21_degree64.bin /data/szr/dataset/sift1b/gnd/idx_100M.ivecs 32 1 10 100000000 128 64 20971520 8 128 10000 4096 >> beam_width_1_RVQ.txt
# sudo ./query /mnt/Samsung980PRO2TB/szr/starling/bigann_100m_M256_R64_L100_B256/_disk.index /home/ErHa/graph_sift1m/bigann_query.bvecs /data/szr/dataset/sift1b/graph/finalgraph/SIFT1B_11_21_degree64.bin /data/szr/dataset/sift1b/gnd/idx_100M.ivecs 64 1 10 100000000 128 64 20971520 8 128 10000 4096 >> beam_width_1_RVQ.txt
# sudo ./query /mnt/Samsung980PRO2TB/szr/starling/bigann_100m_M256_R64_L100_B256/_disk.index /home/ErHa/graph_sift1m/bigann_query.bvecs /data/szr/dataset/sift1b/graph/finalgraph/SIFT1B_11_21_degree64.bin /data/szr/dataset/sift1b/gnd/idx_100M.ivecs 128 1 10 100000000 128 64 20971520 8 128 10000 4096 >> beam_width_1_RVQ.txt
# sudo ./query /mnt/Samsung980PRO2TB/szr/starling/bigann_100m_M256_R64_L100_B256/_disk.index /home/ErHa/graph_sift1m/bigann_query.bvecs /data/szr/dataset/sift1b/graph/finalgraph/SIFT1B_11_21_degree64.bin /data/szr/dataset/sift1b/gnd/idx_100M.ivecs 256 1 10 100000000 128 64 20971520 8 128 10000 4096 >> beam_width_1_RVQ.txt
# sudo ./query /mnt/Samsung980PRO2TB/szr/starling/bigann_100m_M256_R64_L100_B256/_disk.index /home/ErHa/graph_sift1m/bigann_query.bvecs /data/szr/dataset/sift1b/graph/finalgraph/SIFT1B_11_21_degree64.bin /data/szr/dataset/sift1b/gnd/idx_100M.ivecs 512 1 10 100000000 128 64 20971520 8 128 10000 4096 >> beam_width_1_RVQ.txt
# sudo ./query /mnt/Samsung980PRO2TB/szr/starling/bigann_100m_M256_R64_L100_B256/_disk.index /home/ErHa/graph_sift1m/bigann_query.bvecs /data/szr/dataset/sift1b/graph/finalgraph/SIFT1B_11_21_degree64.bin /data/szr/dataset/sift1b/gnd/idx_100M.ivecs 1024 1 10 100000000 128 64 20971520 8 128 10000 4096 >> beam_width_1_RVQ.txt
# sudo ./build /data/szr/dataset/sift1b/graph/subdata/ /data/szr/dataset/sift1b/graph/subgraph/ /data/szr/dataset/sift1b/graph/finalgraph/ 128 32 1000000000 
# sudo ./build /home/ErHa/SIFT1B/SIFT1B.fvecs /data/szr/dataset/sift1b/graph/subgraph/ /data/szr/dataset/sift1b/graph/finalgraph/ 128 32 40000000 
# sudo ./query /data/szr/diskann/sift1b/SIFT1B_DATASET.bin /home/ErHa/graph_sift1m/bigann_query.bvecs /data/szr/dataset/sift1b/graph/finalgraph/SIFT1B_11_21_degree64.bin /data/szr/dataset/sift1b/gnd/idx_100M.ivecs 16 1 10 100000000 128 64
cd ../
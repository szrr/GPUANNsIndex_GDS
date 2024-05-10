#include <vector>

enum metricType{
    L2 = 0,
    IP
};

class GpuIVFFlat {
public:
    GpuIVFFlat(int d, int nlist);
    ~GpuIVFFlat();

    void train(const float* data, int num);
    void add(const float* data, int num);
    void search(const float* queries, int numQueries, int k, float* distances, int* indices);

private:
    int dimension;
    int numLists;
    metricType metric;
    // 假设使用float类型的向量
    std::vector<float> centroids; // 中心向量
    std::vector<std::vector<float>> invertedLists; // 倒排列表

    void computeCentroids(const float* data, int num);
    int assignToClosestCentroid(const float* vec);
};
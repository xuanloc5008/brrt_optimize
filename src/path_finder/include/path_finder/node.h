#ifndef _NODE_H_
#define _NODE_H_

// #include <ros/ros.h>
#include <Eigen/Eigen>
#include <utility>

#include <unordered_map>
#include <queue>
#include <tuple>
#include <cfloat>
#include <list>
#include <iostream>
#include <functional>

#include <unordered_set>
#include <vector>
#include <limits>
#include <cmath>
#include <numeric> // for accumulate
#include <algorithm>

struct TreeNode
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    TreeNode() : parent(NULL), cost_from_start(DBL_MAX), cost_from_parent(0.0){};
    TreeNode *parent;
    Eigen::Vector3d x;
    double cost_from_start;
    double cost_from_parent;
    double heuristic_to_goal;
    double g_plus_h;
    std::list<TreeNode *> children;
};
typedef TreeNode *RRTNode3DPtr;
typedef std::vector<RRTNode3DPtr, Eigen::aligned_allocator<RRTNode3DPtr>> RRTNode3DPtrVector;
typedef std::vector<TreeNode, Eigen::aligned_allocator<TreeNode>> RRTNode3DVector;

class RRTNodeComparator
{
public:
    bool operator()(RRTNode3DPtr node1, RRTNode3DPtr node2)
    {
        return node1->g_plus_h > node2->g_plus_h;
    }
};

struct NodeWithStatus
{
    NodeWithStatus()
    {
        node_ptr = nullptr;
        is_checked = false;
        is_valid = false;
    };
    NodeWithStatus(const RRTNode3DPtr &n, bool checked, bool valid) : node_ptr(n), is_checked(checked), is_valid(valid){};
    RRTNode3DPtr node_ptr;
    bool is_checked;
    bool is_valid; // the segment from a center, not only the node
};

struct Neighbour
{
    Eigen::Vector3d center;
    std::vector<NodeWithStatus> nearing_nodes;
};

// Normalized key: identifies a pair of TreeNodes with their KDTree* origin
struct NodePairKey {
    TreeNode* node1;
    TreeNode* node2;
    void* tree1;
    void* tree2;

    NodePairKey() : node1(nullptr), node2(nullptr), tree1(nullptr), tree2(nullptr) {}

    NodePairKey(TreeNode* a, void* ta, TreeNode* b, void* tb) {
        if (a < b || (a == b && ta < tb)) {
            node1 = a;
            tree1 = ta;
            node2 = b;
            tree2 = tb;
        } else {
            node1 = b;
            tree1 = tb;
            node2 = a;
            tree2 = ta;
        }
    }

    bool operator==(const NodePairKey& other) const {
        return node1 == other.node1 && node2 == other.node2 &&
               tree1 == other.tree1 && tree2 == other.tree2;
    }
};

// Prob info
struct HeuristicInfo {
    double heuristic;
    double probability;

    HeuristicInfo(double h = std::numeric_limits<double>::infinity(), double p = 0.0)
        : heuristic(h), probability(p) {}
};


// Hash function for NodePairKey
struct NodePairHasher {
    std::size_t operator()(const NodePairKey& k) const {
        // combine hashes (simple xor; ok but you may want a better combiner)
        return std::hash<TreeNode*>()(k.node1) ^
               (std::hash<TreeNode*>()(k.node2) << 1) ^
               (std::hash<void*>()(k.tree1) << 2) ^
               (std::hash<void*>()(k.tree2) << 3);
    }
};

// Min-heap entry
struct HeapEntry {
    NodePairKey key;
    double heuristic;
    double probability;

    bool operator>(const HeapEntry& other) const {
        return heuristic > other.heuristic;
    }
};


// Main cache class

class HeuristicCache {
    private:
        std::unordered_map<NodePairKey, HeuristicInfo, NodePairHasher> cache;
        std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> minHeap;
        std::unordered_map<void*, std::unordered_set<NodePairKey, NodePairHasher>> treeIndex;

        void indexTree(void* treeA, void* treeB, const NodePairKey& key) {
            treeIndex[treeA].insert(key);
            treeIndex[treeB].insert(key);
        }

        void unindexTree(void* treeA, void* treeB, const NodePairKey& key) {
            auto erase_key = [&](void* tree) {
                auto it = treeIndex.find(tree);
                if (it != treeIndex.end()) {
                    it->second.erase(key);
                    if (it->second.empty())
                        treeIndex.erase(it);
                }
            };
            erase_key(treeA);
            erase_key(treeB);
        }
    public:
        std::size_t size() const {
            return cache.size();
        }
    
        // insert or update when new heuristic is smaller
        void insert(TreeNode* a, void* treeA, TreeNode* b, void* treeB, double h) {
            NodePairKey key(a, treeA, b, treeB);
            auto it = cache.find(key);
            if (it == cache.end() || it->second.heuristic > h) {
                cache[key] = HeuristicInfo(h, 0.0); // probability will be updated later
                minHeap.push({key, h, 0.0});
                indexTree(treeA, treeB, key);
            }
        }

    
        bool get(TreeNode* a, void* treeA, TreeNode* b, void* treeB,
                double& outH, double& outP) const {
            NodePairKey key(a, treeA, b, treeB);
            auto it = cache.find(key);
            if (it != cache.end()) {
                outH = it->second.heuristic;
                outP = it->second.probability;
                return true;
            }
            return false;
        }

    
        // getMin: returns the minimum-heuristic pair and its heuristic + probability
        bool getMin(TreeNode*& outA, void*& outTreeA, TreeNode*& outB,
                    void*& outTreeB, double& outH, double& outP) {
            while (!minHeap.empty()) {
                const HeapEntry& top = minHeap.top();
                const NodePairKey& key = top.key;
                auto it = cache.find(key);
                if (it != cache.end() && it->second.heuristic == top.heuristic) {
                    outA = key.node1;
                    outTreeA = key.tree1;
                    outB = key.node2;
                    outTreeB = key.tree2;
                    outH = top.heuristic;
                    outP = it->second.probability;
                    return true;
                }
                minHeap.pop();
            }
            return false;
        }


        // Boltzmann update for all cached heuristics
        // avoids C++17 structured bindings for compatibility with older toolchains
        void updateProbabilities(double temperature = 1.0) {
            if (cache.empty()) return;

            double beta = 1.0 / std::max(temperature, 1e-8);
            std::vector<double> expVals;
            expVals.reserve(cache.size());

            // Compute e^(-β * h)
            for (auto it = cache.begin(); it != cache.end(); ++it) {
                expVals.push_back(std::exp(-beta * it->second.heuristic));
            }

            // Compute partition function (sum)
            double Z = std::accumulate(expVals.begin(), expVals.end(), 0.0);
            if (Z <= 0.0) Z = 1e-8;

            // Normalize and assign probabilities (iterate again)
            auto itCache = cache.begin();
            auto itExp = expVals.begin();
            for (; itCache != cache.end() && itExp != expVals.end(); ++itCache, ++itExp) {
                itCache->second.probability = (*itExp) / Z;
            }
        }

        // update single pair probability (manual override)
        bool updateProbability(TreeNode* a, void* treeA, TreeNode* b, void* treeB, double newProb) {
            NodePairKey key(a, treeA, b, treeB);
            auto it = cache.find(key);
            if (it != cache.end()) {
                it->second.probability = newProb;
                return true;
            }
            return false;
        }

        double getMinHeuristic() const {
            while (!minHeap.empty()) {
                const HeapEntry& top = minHeap.top();
                const NodePairKey& key = top.key;
                auto it = cache.find(key);
                if (it != cache.end() && it->second.heuristic == top.heuristic) {
                    return top.heuristic;
                }
                // If this entry is stale, we cannot pop from const method; just break
                break;
            }
            return std::numeric_limits<double>::infinity();
        }

        // find any pair with heuristic == h
        bool getByH(TreeNode*& outA, void*& outTreeA, TreeNode*& outB, void*& outTreeB, double h) {
            for (const auto& entry : cache) {
                if (entry.second.heuristic == h) {
                    const NodePairKey& key = entry.first;
                    outA = key.node1;
                    outTreeA = key.tree1;
                    outB = key.node2;
                    outTreeB = key.tree2;
                    return true;
                }
            }
            return false;
        }
    
        void remove(TreeNode* a, void* treeA, TreeNode* b, void* treeB) {
            NodePairKey key(a, treeA, b, treeB);
            cache.erase(key);
            unindexTree(treeA, treeB, key);
        }
    
        void clear() {
            cache.clear();
            treeIndex.clear();
            minHeap = decltype(minHeap)();
        }
    
        // returns the minimum pair restricted to two specific trees
        bool getMinByTree(void* treeA, void* treeB, TreeNode*& a, TreeNode*& b, double& outH) {
            TreeNode *minA = nullptr, *minB = nullptr;
            void *minTreeA = nullptr, *minTreeB = nullptr;
            double minH = std::numeric_limits<double>::infinity();
            double minP = 0.0;

            // Call getMin that returns full info
            if (getMin(minA, minTreeA, minB, minTreeB, minH, minP)) {
                if ((minTreeA == treeA && minTreeB == treeB) || (minTreeA == treeB && minTreeB == treeA)) {
                    bool is_direct = (minTreeA == treeA);
                    a = is_direct ? minA : minB;
                    b = is_direct ? minB : minA;
                    outH = minH;
                    return true;
                }
            }

            a = b = nullptr;
            outH = std::numeric_limits<double>::infinity();
            return false;
        }

        // find pair by heuristic and restrict to given trees
        bool getPairsByHeuristic(void* treeA, void* treeB, TreeNode*& a, TreeNode*& b, double& outH){
            for (const auto& entry : cache) {
                const HeuristicInfo &info = entry.second;
                const NodePairKey &key = entry.first;
                if (info.heuristic == outH) {
                    if ((key.tree1 == treeA && key.tree2 == treeB) || (key.tree1 == treeB && key.tree2 == treeA)) {
                        a = key.node1;
                        b = key.node2;
                        return true;
                    }
                }
            }
            a = b = nullptr;
            outH = std::numeric_limits<double>::infinity();
            return false;
        }
    
        // pop the minimum pair restriction to given trees
        bool popMinByTree(void* treeA, void* treeB, TreeNode*& outA, TreeNode*& outB, double& outH) {
            while (!minHeap.empty()) {
                const HeapEntry& top = minHeap.top();
                const NodePairKey& key = top.key;
                auto it = cache.find(key);
                if (it == cache.end() || it->second.heuristic != top.heuristic) {
                    minHeap.pop();
                    continue;
                }
                if ((key.tree1 == treeA && key.tree2 == treeB) || (key.tree1 == treeB && key.tree2 == treeA)) {
                    bool is_direct = (key.tree1 == treeA);
                    outA = is_direct ? key.node1 : key.node2;
                    outB = is_direct ? key.node2 : key.node1;
                    outH = top.heuristic;
                    cache.erase(key);
                    minHeap.pop();
                    unindexTree(key.tree1, key.tree2, key);
                    return true;
                }
                minHeap.pop();
            }
            outA = outB = nullptr;
            outH = std::numeric_limits<double>::infinity();
            return false;
        }
    };
#endif

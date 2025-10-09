/*
Copyright (C) 2022 Hongkai Ye (kyle_yeh@163.com)
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE.
*/
#ifndef _NODE_H_
#define _NODE_H_

#include <ros/ros.h>
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
#include <queue>
#include <vector>
#include <limits>



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
typedef vector<RRTNode3DPtr, Eigen::aligned_allocator<RRTNode3DPtr>> RRTNode3DPtrVector;
typedef vector<TreeNode, Eigen::aligned_allocator<TreeNode>> RRTNode3DVector;

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
	vector<NodeWithStatus> nearing_nodes;
};

// Normalized key: identifies a pair of TreeNodes with their KDTree* origin
struct NodePairKey {
    TreeNode* node1;
    TreeNode* node2;
    void* tree1;
    void* tree2;

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

// Hash function for NodePairKey
struct NodePairHasher {
    std::size_t operator()(const NodePairKey& k) const {
        return std::hash<TreeNode*>()(k.node1) ^
               std::hash<TreeNode*>()(k.node2) ^
               std::hash<void*>()(k.tree1) ^
               std::hash<void*>()(k.tree2);
    }
};

// Min-heap entry
struct HeapEntry {
    NodePairKey key;
    double heuristic;

    bool operator>(const HeapEntry& other) const {
        return heuristic > other.heuristic;
    }
};

// Main cache class
struct GuidePairInfo {
    RRTNode3DPtr nodeA;
    RRTNode3DPtr nodeB;
    double h_value;
    double prob; // Boltzmann probability
};
class HeuristicCache {
    private:
        std::unordered_map<NodePairKey, double, NodePairHasher> cache;
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
                    if (it->second.empty()) {
                        treeIndex.erase(it);
                    }
                }
            };
            erase_key(treeA);
            erase_key(treeB);
        }
    
    public:
        std::size_t size() const {
            return cache.size();
        }

        void insert(TreeNode* a, void* treeA, TreeNode* b, void* treeB, double h) {
            NodePairKey key(a, treeA, b, treeB);
            if (cache.count(key) == 0 || cache[key] > h) {
                cache[key] = h;
                minHeap.push({key, h});
                indexTree(treeA, treeB, key);
            }
        }
    
        bool get(TreeNode* a, void* treeA, TreeNode* b, void* treeB, double& outH) const {
            NodePairKey key(a, treeA, b, treeB);
            auto it = cache.find(key);
            if (it != cache.end()) {
                outH = it->second;
                return true;
            }
            return false;
        }
    
        bool getMin(TreeNode*& outA, void*& outTreeA, TreeNode*& outB, void*& outTreeB, double& outH) {
            while (!minHeap.empty()) {
                const HeapEntry& top = minHeap.top();
                const NodePairKey& key = top.key;
                auto it = cache.find(key);
                if (it != cache.end() && it->second == top.heuristic) {
                    outA = key.node1;
                    outTreeA = key.tree1;
                    outB = key.node2;
                    outTreeB = key.tree2;
                    outH = top.heuristic;
                    return true;
                }
                minHeap.pop();
            }
            return false;
        }
    
        double getMinHeuristic() {
            while (!minHeap.empty()) {
                const HeapEntry& top = minHeap.top();
                const NodePairKey& key = top.key;
                auto it = cache.find(key);
                if (it != cache.end() && it->second == top.heuristic) {
                    return top.heuristic;
                }
                minHeap.pop();
            }
            return std::numeric_limits<double>::infinity();
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
    
        bool getMinByTree(void* treeA, void* treeB, TreeNode*& a, TreeNode*& b, double& outH) {
            TreeNode *minA, *minB;
            void *minTreeA, *minTreeB;
            double minH;
            while (getMin(minA, minTreeA, minB, minTreeB, minH)) {
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
    
        bool popMinByTree(void* treeA, void* treeB, TreeNode*& outA, TreeNode*& outB, double& outH) {
            while (!minHeap.empty()) {
                const HeapEntry& top = minHeap.top();
                const NodePairKey& key = top.key;
                auto it = cache.find(key);
                if (it == cache.end() || it->second != top.heuristic) {
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
        //Bolztmann distribution block
        void getAllHeuristic(std::vector<GuidePairInfo>& pairs) {
            pairs.clear();
            pairs.reserve(cache.size());
            for (const auto& entry : cache) {
                const NodePairKey& key = entry.first;
                double h = entry.second;

                GuidePairInfo info;
                info.nodeA = key.node1;
                info.nodeB = key.node2;
                info.h_value = h;
                info.prob = 0.0; // sẽ được tính sau
                pairs.push_back(info);
            }
        }

        //----------------------------------
    };
#endif

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
#include <map>
#include <queue>
#include <tuple>
#include <cfloat>
#include <list>
#include <iostream>
#include <functional>
#include <random>



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

// ============================================================================
// HeuristicCache — now a FIXED-SIZE (capacity N) cache.
//
// Internal ordering structure: std::multimap<double, NodePairKey> sorted_by_h_
//   - sorted ascending by heuristic value h
//   - sorted_by_h_.begin()          -> BEST pair currently cached  (smallest h)
//   - std::prev(sorted_by_h_.end()) -> WORST pair currently cached (largest h,
//                                       i.e. the "tail" of the ordering)
//
// Every insertion that would push size() beyond max_size_ triggers an
// eviction of the worst (tail) element(s) until size() <= max_size_.
// As a fast-path, a new pair that is already worse than the current worst
// pair is rejected immediately (no insertion, no eviction needed).
//
// All public method names/signatures are kept IDENTICAL to the previous
// implementation so callers (e.g. BRRT_Simple_Case3) do not need changes.
// ============================================================================
class HeuristicCache {
    private:
        // Primary ordered structure: h -> key, ascending. Gives O(log N)
        // insert/erase and O(1) access to both the best (begin) and the
        // worst (tail / prev(end)) entries.
        std::multimap<double, NodePairKey> sorted_by_h_;

        // O(1) existence check / O(1) value lookup, mirrors sorted_by_h_.
        std::unordered_map<NodePairKey, double, NodePairHasher> cache;

        // O(1) average removal from sorted_by_h_ given a key, by storing
        // the multimap iterator for each key.
        std::unordered_map<NodePairKey, std::multimap<double, NodePairKey>::iterator, NodePairHasher> key_to_iter_;

        std::unordered_map<void*, std::unordered_set<NodePairKey, NodePairHasher>> treeIndex;

        std::size_t max_size_; // fixed capacity N (0 = cache disabled)

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

        // Erase one entry (by multimap iterator) from ALL bookkeeping structures.
        void eraseEntry(std::multimap<double, NodePairKey>::iterator it) {
            const NodePairKey& key = it->second;
            unindexTree(key.tree1, key.tree2, key);
            key_to_iter_.erase(key);
            cache.erase(key);
            sorted_by_h_.erase(it);
        }

        // Keep popping the worst (tail) element until size() <= max_size_.
        // "Tail" = std::prev(sorted_by_h_.end()), i.e. the largest-h entry.
        void evictWorstIfNeeded() {
            while (sorted_by_h_.size() > max_size_) {
                eraseEntry(std::prev(sorted_by_h_.end()));
            }
        }

    public:
        explicit HeuristicCache(std::size_t max_size = 5000) : max_size_(max_size) {}

        // Change capacity at runtime; immediately evicts worst entries if the
        // new capacity is smaller than the current size.
        void setMaxSize(std::size_t max_size) {
            max_size_ = max_size;
            evictWorstIfNeeded();
        }

        std::size_t maxSize() const { return max_size_; }

        std::size_t size() const {
            return sorted_by_h_.size();
        }

        // Insert (or improve) a pair's heuristic value, subject to the fixed
        // capacity. If the cache is full and the new pair is worse than the
        // current worst entry, it is rejected outright (fast-path, avoids an
        // insert+evict round trip).
        void insert(TreeNode* a, void* treeA, TreeNode* b, void* treeB, double h) {
            if (max_size_ == 0) return; // cache disabled

            NodePairKey key(a, treeA, b, treeB);

            auto existing_it = key_to_iter_.find(key);
            if (existing_it != key_to_iter_.end()) {
                // Pair already cached: only update if the new value is strictly better.
                if (h >= existing_it->second->first) return;
                sorted_by_h_.erase(existing_it->second);
                auto new_it = sorted_by_h_.emplace(h, key);
                existing_it->second = new_it;
                cache[key] = h;
                return; // size unchanged, no eviction needed
            }

            // New pair. If cache is already at capacity, only accept it if it
            // beats the current worst (tail) entry.
            if (sorted_by_h_.size() >= max_size_) {
                auto worst_it = std::prev(sorted_by_h_.end());
                if (h >= worst_it->first) {
                    return; // not good enough to enter the fixed-size cache
                }
            }

            auto new_it = sorted_by_h_.emplace(h, key);
            key_to_iter_[key] = new_it;
            cache[key] = h;
            indexTree(treeA, treeB, key);

            evictWorstIfNeeded(); // safety net (handles size == max_size_ edge case)
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

        // Peek (non-destructive) at the globally best pair currently cached.
        bool getMin(TreeNode*& outA, void*& outTreeA, TreeNode*& outB, void*& outTreeB, double& outH) {
            if (sorted_by_h_.empty()) return false;
            auto it = sorted_by_h_.begin();
            const NodePairKey& key = it->second;
            outA = key.node1;
            outTreeA = key.tree1;
            outB = key.node2;
            outTreeB = key.tree2;
            outH = it->first;
            return true;
        }

        void remove(TreeNode* a, void* treeA, TreeNode* b, void* treeB) {
            NodePairKey key(a, treeA, b, treeB);
            auto it = key_to_iter_.find(key);
            if (it == key_to_iter_.end()) return;
            eraseEntry(it->second);
        }

        void clear() {
            cache.clear();
            treeIndex.clear();
            sorted_by_h_.clear();
            key_to_iter_.clear();
        }

        // Destructive: finds the best pair belonging to (treeA,treeB) or
        // (treeB,treeA), returns it, AND removes it from the cache — matches
        // the original semantics used by BRRT_Simple_Case3::brrt_optimize().
        bool getMinByTree(void* treeA, void* treeB, TreeNode*& outA, TreeNode*& outB, double& outH) {
            for (auto it = sorted_by_h_.begin(); it != sorted_by_h_.end(); ++it) {
                const NodePairKey& key = it->second;
                if ((key.tree1 == treeA && key.tree2 == treeB) || (key.tree1 == treeB && key.tree2 == treeA)) {
                    bool is_direct = (key.tree1 == treeA);
                    outA = is_direct ? key.node1 : key.node2;
                    outB = is_direct ? key.node2 : key.node1;
                    outH = it->first;
                    eraseEntry(it);
                    return true;
                }
            }
            outH = std::numeric_limits<double>::infinity();
            return false;
        }

        // Kept for API compatibility; identical behavior to getMinByTree
        // (both are destructive "pop" operations in this design).
        bool popMinByTree(void* treeA, void* treeB, TreeNode*& outA, TreeNode*& outB, double& outH) {
            return getMinByTree(treeA, treeB, outA, outB, outH);
        }

        bool getBoltzmannPair(void* treeA, void* treeB, TreeNode*& a, TreeNode*& b, double& outH, double temperature = 1.0) {
            std::vector<NodePairKey> candidates;
            std::vector<double> weights;

            for (const auto& kv : cache) {
                const NodePairKey& key = kv.first;
                double h = kv.second;

                bool relevant = (key.tree1 == treeA && key.tree2 == treeB) || (key.tree1 == treeB && key.tree2 == treeA);
                if (relevant) {
                    candidates.push_back(key);
                    weights.push_back(std::exp(-h / temperature));
                }
            }

            if (candidates.empty()) return false;

            std::discrete_distribution<> dist(weights.begin(), weights.end());
            static std::random_device rd;
            static std::mt19937 gen(rd());

            int idx = dist(gen);
            const NodePairKey& selected = candidates[idx];

            bool is_direct = (selected.tree1 == treeA);
            a = is_direct ? selected.node1 : selected.node2;
            b = is_direct ? selected.node2 : selected.node1;
            outH = cache.at(selected);

            return true;
        }

        void removeNodesInside(const Eigen::Vector3d& center, double radius, void* treeA, void* treeB) {
            double r_sq = radius * radius;
            auto it = sorted_by_h_.begin();
            while (it != sorted_by_h_.end()) {
                const NodePairKey key = it->second; // copy: `it` is invalidated below
                bool relevant = (key.tree1 == treeA && key.tree2 == treeB) || (key.tree1 == treeB && key.tree2 == treeA);

                if (relevant) {
                    double d1 = (key.node1->x - center).squaredNorm();
                    double d2 = (key.node2->x - center).squaredNorm();

                    if (d1 < r_sq || d2 < r_sq) {
                        auto to_erase = it;
                        ++it; // advance before invalidating to_erase
                        unindexTree(key.tree1, key.tree2, key);
                        key_to_iter_.erase(key);
                        cache.erase(key);
                        sorted_by_h_.erase(to_erase);
                        continue;
                    }
                }
                ++it;
            }
        }
    };
#endif

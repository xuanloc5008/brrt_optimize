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
#include <vector>
#include <limits>
#include <cmath>
#include <numeric> // for accumulate
#include <algorithm>
#include <random>

namespace path_plan { // Assuming namespace path_plan based on brrt_simple_case2.h

struct TreeNode
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    TreeNode() : parent(nullptr), cost_from_start(std::numeric_limits<double>::infinity()), cost_from_parent(0.0), heuristic_to_goal(0.0), g_plus_h(std::numeric_limits<double>::infinity()) {}; // Initialize members properly
    TreeNode *parent;
    Eigen::Vector3d x;
    double cost_from_start;
    double cost_from_parent;
    double heuristic_to_goal; // Not used in HeuristicCache but part of node
    double g_plus_h;         // Not used in HeuristicCache but part of node
    std::list<TreeNode *> children;
};
typedef TreeNode *RRTNode3DPtr;
typedef std::vector<RRTNode3DPtr, Eigen::aligned_allocator<RRTNode3DPtr>> RRTNode3DPtrVector;
typedef std::vector<TreeNode, Eigen::aligned_allocator<TreeNode>> RRTNode3DVector;

// Structure for HeuristicCache::getAllEntries return type
struct HeuristicEntryData {
    TreeNode* node1;
    TreeNode* node2;
    void* tree1; // Pointer to the kdtree node1 belongs to
    void* tree2; // Pointer to the kdtree node2 belongs to
    double heuristicValue;
};


class RRTNodeComparator
{
public:
    bool operator()(RRTNode3DPtr node1, RRTNode3DPtr node2)
    {
        // Compare g_plus_h, handle potential infinity values if necessary
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
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW; // Add if Eigen members are used directly
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

    // Normalize order based on node pointer address first, then tree pointer address
    NodePairKey(TreeNode* a, void* ta, TreeNode* b, void* tb) {
        if (reinterpret_cast<uintptr_t>(a) < reinterpret_cast<uintptr_t>(b) ||
           (a == b && reinterpret_cast<uintptr_t>(ta) < reinterpret_cast<uintptr_t>(tb)))
        {
            node1 = a; tree1 = ta; node2 = b; tree2 = tb;
        } else {
            node1 = b; tree1 = tb; node2 = a; tree2 = ta;
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
    double probability; // Note: This might not be actively used/updated by default logic

    HeuristicInfo(double h = std::numeric_limits<double>::infinity(), double p = 0.0)
        : heuristic(h), probability(p) {}
};


// Hash function for NodePairKey
struct NodePairHasher {
    std::size_t operator()(const NodePairKey& k) const {
        // More robust hash combining
        std::size_t seed = 0;
        // Combine hashes using a sequence recommended by Boost's hash_combine
        seed ^= std::hash<TreeNode*>()(k.node1) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<TreeNode*>()(k.node2) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<void*>()(k.tree1) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<void*>()(k.tree2) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

// Min-heap entry
struct HeapEntry {
    NodePairKey key;
    double heuristic;
    // double probability; // Probability is stored in the map, maybe not needed here

    // Comparison for min-heap based on heuristic value
    bool operator>(const HeapEntry& other) const {
        return heuristic > other.heuristic;
    }
};


// Main cache class
class HeuristicCache {
    private:
        std::unordered_map<NodePairKey, HeuristicInfo, NodePairHasher> cache;
        // Min-heap stores HeapEntry objects, uses std::vector as container, compares using std::greater
        std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> minHeap;
        // Index mapping tree pointers to the set of keys involving that tree
        std::unordered_map<void*, std::unordered_set<NodePairKey, NodePairHasher>> treeIndex;

        // Helper to add a key to the index for both trees involved
        void indexTree(void* treeA, void* treeB, const NodePairKey& key) {
            if (treeA) treeIndex[treeA].insert(key);
            if (treeB) treeIndex[treeB].insert(key);
        }

        // Helper to remove a key from the index for both trees involved
        void unindexTree(void* treeA, void* treeB, const NodePairKey& key) {
            auto erase_key = [&](void* tree) {
                if (!tree) return; // Skip null tree pointers
                auto it = treeIndex.find(tree);
                if (it != treeIndex.end()) {
                    it->second.erase(key);
                    // Remove the tree entry itself if its set becomes empty
                    if (it->second.empty())
                        treeIndex.erase(it);
                }
            };
            erase_key(treeA);
            erase_key(treeB);
        }
    public:
        // Returns the number of heuristic pairs currently stored
        std::size_t size() const {
            return cache.size();
        }
        // NEW: Structure to hold pair info for sorting
        struct HeuristicPairInfo {
            TreeNode* nodeA; // Node from treeA
            TreeNode* nodeB; // Node from treeB
            double heuristic;

            // Comparison for sorting (ascending by heuristic)
            bool operator<(const HeuristicPairInfo& other) const {
                return heuristic < other.heuristic;
            }
        };

        // NEW: Function to get all pairs between two specific trees
        std::vector<HeuristicPairInfo> getAllPairsByTree(void* treeA, void* treeB) {
            std::vector<HeuristicPairInfo> pairs;
            if (!treeA || !treeB) return pairs;

            auto it_index_A = treeIndex.find(treeA);
            if (it_index_A != treeIndex.end()) {
                for (const NodePairKey& key : it_index_A->second) {
                    // Check if this key involves treeB as the *other* tree
                    void* other_tree = (key.tree1 == treeA) ? key.tree2 : key.tree1;
                    if (other_tree == treeB) {
                        auto it_cache = cache.find(key);
                        // Ensure the entry still exists in the cache
                        if (it_cache != cache.end()) {
                            if (key.tree1 == treeA) {
                                pairs.push_back({key.node1, key.node2, it_cache->second.heuristic});
                            } else {
                                // key.tree1 == treeB, so key.node1 is from treeB
                                // key.tree2 == treeA, so key.node2 is from treeA
                                pairs.push_back({key.node2, key.node1, it_cache->second.heuristic});
                            }
                        }
                    }
                }
            }
            return pairs;
        }
        // ***** CORRECTED IMPLEMENTATION (defined inside class) *****
        std::vector<HeuristicEntryData> getAllEntries() const {
            std::vector<HeuristicEntryData> entries;
            if (cache.empty()) {
                return entries; // Return empty vector if cache is empty
            }
            entries.reserve(cache.size());
            for (const auto& pair : cache) { // Iterate through the unordered_map
                const NodePairKey& key = pair.first;
                const HeuristicInfo& info = pair.second;
                // Add check for nullptrs before adding
                if (key.node1 && key.node2 && key.tree1 && key.tree2) {
                   entries.push_back({key.node1, key.node2, key.tree1, key.tree2, info.heuristic});
                } else {
                   // Optional: Log a warning if invalid entries are found
                   // ROS_WARN_THROTTLE(5.0, "getAllEntries: Found cache entry with null pointers.");
                }
            }
            return entries;
        }
        // **********************************************************


        // Inserts or updates a heuristic value for a pair of nodes.
        // Updates only if the new heuristic 'h' is smaller than the existing one.
        void insert(TreeNode* a, void* treeA, TreeNode* b, void* treeB, double h) {
            if (!a || !b || !treeA || !treeB) return; // Basic validation
            NodePairKey key(a, treeA, b, treeB);
            auto it = cache.find(key);
            // Insert if not found, or update if new heuristic is better (smaller)
            if (it == cache.end() || h < it->second.heuristic) {
                // Store/update in the map. Probability is not set here.
                cache[key] = HeuristicInfo(h, 0.0);
                // Push the new best heuristic onto the min-heap
                minHeap.push({key, h}); // Probability not needed in heap entry
                // Update the tree index
                indexTree(treeA, treeB, key);
            }
        }


        // Retrieves the heuristic and probability for a specific node pair.
        bool get(TreeNode* a, void* treeA, TreeNode* b, void* treeB,
                double& outH, double& outP) const {
            if (!a || !b || !treeA || !treeB) return false;
            NodePairKey key(a, treeA, b, treeB);
            auto it = cache.find(key);
            if (it != cache.end()) {
                outH = it->second.heuristic;
                outP = it->second.probability; // Returns stored probability
                return true;
            }
            // Default values if not found
            outH = std::numeric_limits<double>::infinity();
            outP = 0.0;
            return false;
        }

        /**
         * @brief Prints the heuristic and probability of all entries
         * in the cache to the ROS log.
         */
        void printCacheContents() const
        {
            // Make sure you have #include <ros/console.h> at the top of this file
            if (cache.empty()) {
                ROS_INFO_THROTTLE(1.0, "[HeuristicCache] Cache is EMPTY.");
                return;
            }

            ROS_INFO("--- Heuristic Cache Contents (%lu entries) ---", cache.size());
            
            // Iterate through the map
            // 'entry' is a std::pair<const NodePairKey, HeuristicInfo>
            for (const auto& entry : cache) {
                const HeuristicInfo& info = entry.second;
                ROS_INFO("  H: %-10.4f  P: %.6f", info.heuristic, info.probability);
            }
            ROS_INFO("-----------------------------------------------------");
        }
        // Retrieves the overall minimum heuristic pair from the cache efficiently using the min-heap.
        // Handles stale entries in the heap.
        bool getMin(TreeNode*& outA, void*& outTreeA, TreeNode*& outB,
                    void*& outTreeB, double& outH, double& outP) {
            // Clean up stale entries from the top of the heap
            while (!minHeap.empty()) {
                const HeapEntry& top = minHeap.top();
                const NodePairKey& key = top.key;
                auto it = cache.find(key);
                // Check if the entry still exists in the cache and has the same heuristic value
                if (it != cache.end() && std::abs(it->second.heuristic - top.heuristic) < 1e-9) {
                    // Found a valid minimum entry
                    outA = key.node1;       outTreeA = key.tree1;
                    outB = key.node2;       outTreeB = key.tree2;
                    outH = top.heuristic;
                    outP = it->second.probability; // Get probability from map
                    return true;
                }
                // Entry is stale (removed from cache or updated with a different value), pop it
                minHeap.pop();
            }
            // Heap is empty or only contained stale entries
            outH = std::numeric_limits<double>::infinity();
            outP = 0.0;
            return false;
        }

        // --- updateProbabilities might not be needed if Boltzmann selection computes weights on demand ---
        // (Keep if you have other uses for pre-calculated probabilities)
        /*
        void updateProbabilities(double temperature = 1.0) {
           // ... implementation as provided ...
        }
        */
        // --- NEW FUNCTION: updateAllProbabilities ---
        /**
         * @brief Recalculates the Boltzmann (softmax) probability for all entries in the cache.
         * Lower heuristics get higher probabilities.
         * @param temperature The Boltzmann temperature (T).
         * - T -> 0  : Becomes greedy (selects min_h)
         * - T -> inf: Becomes uniform random
         */
        void updateAllProbabilities(double temperature) {
            if (cache.empty() || temperature <= 0.0) {
                // Set all to 0 if cache is empty or T is invalid
                for (auto& entry : cache) {
                    entry.second.probability = 0.0;
                }
                return;
            }

            double min_h = std::numeric_limits<double>::infinity();
            // Find min heuristic for stable softmax (shift-invariance)
            for (const auto& entry : cache) {
                if (std::isfinite(entry.second.heuristic)) {
                    min_h = std::min(min_h, entry.second.heuristic);
                }
            }
            
            // Handle case where all heuristics are infinity
            if (!std::isfinite(min_h)) {
                min_h = 0.0; // Set to 0, all exp() will be 0
            }

            double sum_of_exps = 0.0;
            // Calculate sum of exponentials
            // We use -h because we want *low* heuristics to have *high* probability
            for (const auto& entry : cache) {
                if (std::isfinite(entry.second.heuristic)) {
                    sum_of_exps += std::exp(-(entry.second.heuristic - min_h) / temperature);
                }
            }

            // Assign probabilities
            if (sum_of_exps == 0.0) {
                // This happens if cache is empty or all heuristics are infinity
                // Fallback to uniform probability
                double uniform_prob = 1.0 / cache.size();
                for (auto& entry : cache) {
                    entry.second.probability = uniform_prob;
                }
            } else {
                // Assign Boltzmann probability
                for (auto& entry : cache) {
                    if (std::isfinite(entry.second.heuristic)) {
                        entry.second.probability = std::exp(-(entry.second.heuristic - min_h) / temperature) / sum_of_exps;
                    } else {
                        entry.second.probability = 0.0; // Inf heuristic gets 0 probability
                    }
                }
            }
        }
        // --- END NEW FUNCTION ---

        // --- NEW FUNCTION: popByProbability ---
        /**
         * @brief Probabilistically selects and removes a pair between treeA and treeB
         * based on the pre-calculated probabilities in the cache.
         * @param treeA The first tree
         * @param treeB The second tree
         * @param random_roll A random double, expected to be in [0.0, 1.0).
         * @param outNodeA Output for node from treeA
         * @param outNodeB Output for node from treeB
         * @param outH Output for the selected pair's heuristic
         * @return true if a pair was selected and popped, false otherwise (e.g., no pairs exist)
         */
        bool popByProbability(void* treeA, void* treeB, double random_roll,
                              TreeNode*& outNodeA, TreeNode*& outNodeB, double& outH) {
            if (!treeA || !treeB) return false;

            // 1. Find all relevant entries and their total probability sum
            std::vector<NodePairKey> relevant_keys;
            double relevant_prob_sum = 0.0;
            
            auto it_index_A = treeIndex.find(treeA);
            if (it_index_A == treeIndex.end()) {
                return false; // No entries for treeA
            }

            for (const NodePairKey& key : it_index_A->second) {
                void* other_tree = (key.tree1 == treeA) ? key.tree2 : key.tree1;
                if (other_tree == treeB) {
                    auto it_cache = cache.find(key);
                    if (it_cache != cache.end()) {
                        relevant_keys.push_back(key);
                        relevant_prob_sum += it_cache->second.probability;
                    }
                }
            }

            if (relevant_keys.empty() || relevant_prob_sum <= 0.0) {
                // No pairs found or total probability is zero
                return false;
            }

            // 2. Perform roulette wheel selection
            double r = random_roll * relevant_prob_sum; // Scale roll to the sum
            double cumulative_prob = 0.0;
            NodePairKey selected_key;
            bool selected = false;

            for (const NodePairKey& key : relevant_keys) {
                cumulative_prob += cache[key].probability; // Use [] since we know key exists
                if (r <= cumulative_prob) {
                    selected_key = key;
                    selected = true;
                    break;
                }
            }

            // Handle potential floating point errors (select last one)
            if (!selected && !relevant_keys.empty()) {
                selected_key = relevant_keys.back();
            }

            // 3. We have a selection. Get data, pop it, and return.
            const HeuristicInfo& info = cache[selected_key];
            outH = info.heuristic;

            // Assign output nodes based on which tree is treeA
            if (selected_key.tree1 == treeA) {
                outNodeA = selected_key.node1;
                outNodeB = selected_key.node2;
            } else {
                outNodeA = selected_key.node2;
                outNodeB = selected_key.node1;
            }

            // 4. Pop the entry
            cache.erase(selected_key);
            unindexTree(treeA, treeB, selected_key);
            // Note: Heap entry becomes stale, will be handled by getMin

            return true;
        }
        // --- END NEW FUNCTION ---
        // update single pair probability (manual override) - Keep if needed
        bool updateProbability(TreeNode* a, void* treeA, TreeNode* b, void* treeB, double newProb) {
            if (!a || !b || !treeA || !treeB) return false;
            NodePairKey key(a, treeA, b, treeB);
            auto it = cache.find(key);
            if (it != cache.end()) {
                it->second.probability = newProb;
                return true;
            }
            return false;
        }
        // Returns the smallest heuristic value in the cache if not empty.
        // Returns +∞ if cache is empty or only contains stale entries.
        double getLowestHeuristicIfNotEmpty() {
            if (cache.empty()) {
                return std::numeric_limits<double>::infinity(); // cache is empty
            }

            // Clean up stale entries from the top of the heap
            while (!minHeap.empty()) {
                const HeapEntry& top = minHeap.top();
                const NodePairKey& key = top.key;
                auto it = cache.find(key);

                // Found valid (non-stale) entry
                if (it != cache.end() && std::abs(it->second.heuristic - top.heuristic) < 1e-9) {
                    return top.heuristic;
                }

                // Otherwise, remove stale entry
                minHeap.pop();
            }

            // If heap is empty or only had stale entries
            return std::numeric_limits<double>::infinity();
        }



        // Returns the overall minimum heuristic value currently in the cache.
        double getMinHeuristic() { // Made non-const to allow popping stale entries
            // Clean up stale entries from the top of the heap
            while (!minHeap.empty()) {
                const HeapEntry& top = minHeap.top();
                const NodePairKey& key = top.key;
                auto it = cache.find(key);
                // Check if the entry still exists and has the expected heuristic
                if (it != cache.end() && std::abs(it->second.heuristic - top.heuristic) < 1e-9) {
                    return top.heuristic; // Found the valid minimum
                }
                // Pop stale entry
            }
            // Heap is empty or only contained stale entries
            return 0.0;
        }

        // Finds *a* pair with a specific heuristic value 'h'. Not guaranteed to be unique.
        bool getByH(TreeNode*& outA, void*& outTreeA, TreeNode*& outB, void*& outTreeB, double h) const {
             if (!std::isfinite(h)) return false;
            for (const auto& entry : cache) {
                // Use tolerance comparison for floating point values
                if (std::abs(entry.second.heuristic - h) < 1e-9) {
                    const NodePairKey& key = entry.first;
                    // Check if pointers are valid before assigning
                    if (key.node1 && key.node2 && key.tree1 && key.tree2) {
                        outA = key.node1; outTreeA = key.tree1;
                        outB = key.node2; outTreeB = key.tree2;
                        return true;
                    }
                }
            }
            return false;
        }

        // Removes a specific node pair from the cache and index.
        void remove(TreeNode* a, void* treeA, TreeNode* b, void* treeB) {
             if (!a || !b || !treeA || !treeB) return;
            NodePairKey key(a, treeA, b, treeB);
            // Remove from map first
            if (cache.erase(key) > 0) {
                 // Remove from tree index if it was present in the map
                 unindexTree(treeA, treeB, key);
                 // Note: Corresponding entries in minHeap become stale and are handled by getMin/getMinHeuristic
            }
        }

        // Clears the entire cache, heap, and index.
        void clear() {
            cache.clear();
            treeIndex.clear();
            // Replace the priority queue with a new empty one
            minHeap = std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>>();
        }


        // Retrieves the minimum heuristic pair specifically between treeA and treeB.
        // This implementation iterates through indexed entries, which might be less efficient than using the heap
        // if the number of entries per tree is large.
        bool getMinByTree(void* treeA, void* treeB, TreeNode*& outNodeA, TreeNode*& outNodeB, double& outH) {
            if (!treeA || !treeB) return false;

            double min_h = std::numeric_limits<double>::infinity();
            bool found = false;
            const NodePairKey* best_key_ptr = nullptr;

            // Iterate through keys associated with treeA (potentially smaller set)
            auto it_index_A = treeIndex.find(treeA);
            if (it_index_A != treeIndex.end()) {
                for (const NodePairKey& key : it_index_A->second) {
                    // Check if this key involves treeB as the *other* tree
                    void* other_tree = (key.tree1 == treeA) ? key.tree2 : key.tree1;
                    if (other_tree == treeB) {
                        auto it_cache = cache.find(key);
                        // Ensure the entry still exists in the cache
                        if (it_cache != cache.end()) {
                            if (it_cache->second.heuristic < min_h) {
                                min_h = it_cache->second.heuristic;
                                best_key_ptr = &key; // Store pointer to the best key found so far
                                found = true;
                            }
                        }
                    }
                }
            }

            if (found && best_key_ptr) {
                // Assign output nodes based on which tree is treeA
                if (best_key_ptr->tree1 == treeA) {
                    outNodeA = best_key_ptr->node1;
                    outNodeB = best_key_ptr->node2;
                } else {
                    outNodeA = best_key_ptr->node2; // node2 is from treeA
                    outNodeB = best_key_ptr->node1; // node1 is from treeB
                }
                outH = min_h;
                return true;
            }

            // If not found (or index was empty)
            outNodeA = outNodeB = nullptr;
            outH = std::numeric_limits<double>::infinity();
            return false;
        }


        // Finds *a* pair with heuristic 'outH' between treeA and treeB.
        bool getPairsByHeuristic(void* treeA, void* treeB, TreeNode*& outNodeA, TreeNode*& outNodeB, double targetH){
            if (!treeA || !treeB || !std::isfinite(targetH)) return false;

            // Iterate through keys associated with treeA
             auto it_index_A = treeIndex.find(treeA);
             if (it_index_A != treeIndex.end()) {
                 for (const NodePairKey& key : it_index_A->second) {
                     // Check if the other tree is treeB
                     void* other_tree = (key.tree1 == treeA) ? key.tree2 : key.tree1;
                     if (other_tree == treeB) {
                         auto it_cache = cache.find(key);
                         // Check if entry exists and heuristic matches (with tolerance)
                         if (it_cache != cache.end() && std::abs(it_cache->second.heuristic - targetH) < 1e-9) {
                             // Assign output nodes based on which tree is treeA
                             if (key.tree1 == treeA) {
                                 outNodeA = key.node1; outNodeB = key.node2;
                             } else {
                                 outNodeA = key.node2; outNodeB = key.node1;
                             }
                             return true; // Found a matching pair
                         }
                     }
                 }
             }
             // No matching pair found
            outNodeA = outNodeB = nullptr;
            return false;
        }

        // Pops (removes and returns) the minimum heuristic pair between treeA and treeB.
        // Requires iterating through the heap to find the specific pair, less efficient.
        bool popMinByTree(void* treeA, void* treeB, TreeNode*& outNodeA, TreeNode*& outNodeB, double& outH) {
             if (!treeA || !treeB) return false;

             // Find the minimum pair between these trees first (using existing method)
             if (!getMinByTree(treeA, treeB, outNodeA, outNodeB, outH)) {
                 return false; // No pair exists between these trees
             }

             // Now remove the found pair
             // Reconstruct the key (normalization ensures consistency)
             NodePairKey key_to_remove(outNodeA, treeA, outNodeB, treeB);
             if (cache.erase(key_to_remove) > 0) {
                 unindexTree(treeA, treeB, key_to_remove);
                 // Note: Heap entry becomes stale, handled by getMin/getMinHeuristic later
                 return true;
             } else {
                 // Should not happen if getMinByTree succeeded
                 ROS_ERROR("popMinByTree: Failed to erase key after getMinByTree found it!");
                 return false;
             }
        }
    }; // End class HeuristicCache

} // End namespace path_plan

#endif // _NODE_H_
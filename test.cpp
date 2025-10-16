#include "src/path_finder/include/path_finder/node.h"
#include <iostream>
#include <random>

int main() {
    HeuristicCache cache;

    // Create a few fake TreeNodes (they just need distinct pointers)
    TreeNode n1, n2, n3, n4, n5;
    void* treeA = reinterpret_cast<void*>(0xAAAA);
    void* treeB = reinterpret_cast<void*>(0xBBBB);

    // Insert some sample heuristic values
    cache.insert(&n1, treeA, &n2, treeB, 2.5);
    cache.insert(&n3, treeA, &n4, treeB, 1.2);
    cache.insert(&n2, treeA, &n5, treeB, 4.0);
    cache.insert(&n1, treeA, &n3, treeB, 0.8);

    std::cout << "Cache size after inserts: " << cache.size() << "\n";

    // Compute Boltzmann probabilities with temperature = 1.0
    cache.updateProbabilities(1.0);

    // Display all stored pairs and their probabilities
    std::cout << "\n--- Cache contents ---\n";
    for (double h : {0.8, 1.2, 2.5, 4.0}) {
        TreeNode* a; void* ta; TreeNode* b; void* tb;
        double H, P;
        if (cache.getByH(a, ta, b, tb, h)) {
            cache.get(a, ta, b, tb, H, P);
            std::cout << "Heuristic: " << H 
                      << "  Probability: " << P
                      << "  (nodes: " << a << ", " << b << ")\n";
        }
    }

    // Retrieve minimum
    TreeNode* minA; void* tA; TreeNode* minB; void* tB;
    double minH, minP;
    if (cache.getMin(minA, tA, minB, tB, minH, minP)) {
        std::cout << "\nMin heuristic pair: " << minA << " - " << minB
                  << "  H=" << minH << "  P=" << minP << "\n";
    }

    // Pop the minimum and re-check
    TreeNode* popA; TreeNode* popB; double popH;
    if (cache.popMinByTree(treeA, treeB, popA, popB, popH)) {
        std::cout << "Popped min pair: " << popA << " - " << popB 
                  << "  H=" << popH << "\n";
    }

    std::cout << "\nCache size after pop: " << cache.size() << "\n";

    // Clear cache
    cache.clear();
    std::cout << "Cache cleared. Final size: " << cache.size() << "\n";

    return 0;
}

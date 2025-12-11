/*
Copyright (C) 2022 Hongkai Ye (kyle_yeh@163.com), Longji Yin (ljyin6038@163.com )
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
#ifndef BRRT_OPTIMIZE_CASE3_H
#define BRRT_OPTIMIZE_CASE3_H

#include "occ_grid/occ_map.h"
#include "visualization/visualization.hpp"
#include "sampler.h"
#include "node.h"
#include "kdtree.h"

#include <ros/ros.h>
#include <utility>
#include <queue>
#include <algorithm>
namespace path_plan
{
  class BRRT_Optimize_Case3
  {
  public:
  BRRT_Optimize_Case3() {};
  BRRT_Optimize_Case3(const ros::NodeHandle &nh, const env::OccMap::Ptr &mapPtr) : nh_(nh), map_ptr_(mapPtr)
    {
      nh_.param("BRRT_Optimize/h_threshold", h_threshold_, -0.01);         
      nh_.param("BRRT_Optimize/trap_count_limit", trap_count_limit_, 50); 
      nh_.param("BRRT_Optimize/trap_step_limit", trap_step_limit_, 50);

      nh_.param("BRRT/steer_length", steer_length_, 0.0);
      nh_.param("BRRT/search_time", search_time_, 0.0);
      nh_.param("BRRT/max_tree_node_nums", max_tree_node_nums_, 0);

      nh_.param("BRRT_Optimize/p1", brrt_optimize_p1_, 0.8);
      nh_.param("BRRT_Optimize/u_p", brrt_optimize_u_p, 1.0);
      nh_.param("BRRT_Optimize/step", brrt_optimize_step_, 0.1);

      nh_.param("BRRT_Optimize/alpha", brrt_optimize_alpha_, 0.5);
      nh_.param("BRRT_Optimize/beta", brrt_optimize_beta_, 0.3);
      nh_.param("BRRT_Optimize/gamma", brrt_optimize_gamma_, 0.5);
      nh_.param("BRRT_Optimize/max_iteration", max_iteration_, 0);
      nh_.param("BRRT_Optimize/enable2d", brrt_enable_2d, true);
      nh_.param("BRRT_Optimize/rewire_radius", rewire_radius_, 2.0); // Radius to search for neighbors
      nh_.param("BRRT_Optimize/animation_delay", animation_delay_, 0.00); // Default 0.05 seconds

      ROS_WARN_STREAM("[BRRT_Optimize_Case3] param: steer_length: " << steer_length_);
      ROS_WARN_STREAM("[BRRT_Optimize_Case3] param: search_time: " << search_time_);
      ROS_WARN_STREAM("[BRRT_Optimize_Case3] param: max_tree_node_nums: " << max_tree_node_nums_);
      

      sampler_.setSamplingRange(mapPtr->getOrigin(), mapPtr->getMapSize());

      valid_tree_node_nums_ = 0;
      nodes_pool_.resize(max_tree_node_nums_);
      for (int i = 0; i < max_tree_node_nums_; ++i)
      {
        nodes_pool_[i] = new TreeNode;
      }
    }
    ~BRRT_Optimize_Case3() {};

    bool plan(const Eigen::Vector3d &s, const Eigen::Vector3d &g)
    {
      reset();
      /* construct start and goal nodes */
      start_node_ = nodes_pool_[1];
      start_node_->x = s;
      start_node_->cost_from_start = 0.0;
      goal_node_ = nodes_pool_[0];
      goal_node_->x = g;
      goal_node_->cost_from_start = 0.0; // important
      valid_tree_node_nums_ = 2;         // put start and goal in tree

      // vis_ptr_->visualize_a_ball(s, 0.3, "start", visualization::Color::pink);
      // vis_ptr_->visualize_a_ball(g, 0.3, "goal", visualization::Color::steelblue);
      cache.clear(); // clear the heuristic cache before planning
      return brrt_optimize(s, g);
    }

    vector<Eigen::Vector3d> getPath()
    {
      return final_path_;
    }

    vector<vector<Eigen::Vector3d>> getAllPaths()
    {
      return path_list_;
    }

    vector<std::pair<double, double>> getSolutions()
    {
      return solution_cost_time_pair_list_;
    }
    void set_heuristic_param(double p1, double u_p, double alpha, double beta, double gamma,double steer_length)
    {
      brrt_optimize_p1_ = p1;
      brrt_optimize_u_p = u_p;
      brrt_optimize_alpha_ = alpha;
      brrt_optimize_beta_ = beta;
      brrt_optimize_gamma_ = gamma;
      steer_length_  = steer_length;
    }
    void setVisualizer(const std::shared_ptr<visualization::Visualization> &visPtr)
    {
      vis_ptr_ = visPtr;
    };
    int get_number_of_iteration()
    {
      return number_of_iterations_;
    }
    int get_valid_tree_node_nums()
    {
      return valid_tree_node_nums_;
    }
    double get_final_path_use_time_()
    {
      return final_path_use_time_;
    }
  private:
    // nodehandle params

    ros::NodeHandle nh_;
    double h_threshold_;        // Threshold to determine if heuristic progress is too slow
    int trap_count_limit_;      // How many bad iterations before we say we are "stuck"
    int trap_step_limit_;       // Max iterations allowed inside trap mode
    Eigen::Vector3d trap_center_; // Center of the trap circle
    double trap_radius_ = 2.0;

    BiasSampler sampler_;
    double brrt_optimize_p1_;
    double brrt_optimize_u_p;
    double brrt_optimize_step_;
    double brrt_optimize_alpha_;
    double brrt_optimize_beta_;
    double brrt_optimize_gamma_;
    int max_iteration_;
    double steer_length_;
    double search_time_;
    int max_tree_node_nums_;
    int number_of_iterations_;
    int valid_tree_node_nums_;
    double first_path_use_time_;
    double final_path_use_time_;
    bool brrt_enable_2d;
    double rewire_radius_;
    // SOF-RRT* Sampling Parameters
    double animation_delay_;
    double sof_epsilon_ = 0.4;      // Probability to explore vs exploit [cite: 229]
    double sof_weight_grade_ = 0.6; // Weight power factor [cite: 270]
    int sof_n_blocks_ = 20;         // Number of "Lidar" sectors [cite: 228]
    double sof_lidar_range_ = 5.0;  // Max range for simulated lidar [cite: 268]
    double cost_best_;
    std::vector<TreeNode *> nodes_pool_;
    TreeNode *start_node_;
    TreeNode *goal_node_;
    vector<Eigen::Vector3d> final_path_;
    vector<vector<Eigen::Vector3d>> path_list_;
    vector<std::pair<double, double>> solution_cost_time_pair_list_;

    // environment
    env::OccMap::Ptr map_ptr_;
    std::shared_ptr<visualization::Visualization> vis_ptr_;
    HeuristicCache cache;
    void reset()
    {
      final_path_.clear();
      path_list_.clear();
      cost_best_ = DBL_MAX;

      solution_cost_time_pair_list_.clear();
      for (int i = 0; i < valid_tree_node_nums_; i++)
      {
        nodes_pool_[i]->parent = nullptr;
        nodes_pool_[i]->children.clear();
      }
      valid_tree_node_nums_ = 0;
    }

    double calDist(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2)
    {
      return (p1 - p2).norm();
    }

    RRTNode3DPtr addTreeNode(RRTNode3DPtr parent, const Eigen::Vector3d &state,
                         const double &cost_from_start, const double &cost_from_parent)
    {
      RRTNode3DPtr new_node_ptr = nodes_pool_[valid_tree_node_nums_];
      valid_tree_node_nums_++;
      new_node_ptr->parent = parent;
      parent->children.push_back(new_node_ptr);
      new_node_ptr->x = state;
      new_node_ptr->cost_from_start = cost_from_start;
      new_node_ptr->cost_from_parent = cost_from_parent;
      return new_node_ptr;
    }

    void changeNodeParent(RRTNode3DPtr node, RRTNode3DPtr parent, const double &cost_from_parent)
    {
      if (node->parent)
        node->parent->children.remove(node); // DON'T FORGET THIS, remove it form its parent's children list
      node->parent = parent;
      node->cost_from_parent = cost_from_parent;
      node->cost_from_start = parent->cost_from_start + cost_from_parent;
      parent->children.push_back(node);

      // for all its descedants, change the cost_from_start and tau_from_start;
      RRTNode3DPtr descendant(node);
      std::queue<RRTNode3DPtr> Q;
      Q.push(descendant);
      while (!Q.empty())
      {
        descendant = Q.front();
        Q.pop();
        for (const auto &leafptr : descendant->children)
        {
          leafptr->cost_from_start = leafptr->cost_from_parent + descendant->cost_from_start;
          Q.push(leafptr);
        }
      }
    }

    void fillPath(const RRTNode3DPtr &node_A, const RRTNode3DPtr &node_B, vector<Eigen::Vector3d> &path)
    {
      path.clear();
      RRTNode3DPtr node_ptr = node_A;
      while (node_ptr->parent)
      {
        path.push_back(node_ptr->x);
        node_ptr = node_ptr->parent;
      }
      path.push_back(start_node_->x);
      std::reverse(std::begin(path), std::end(path));

      node_ptr = node_B;
      while (node_ptr->parent)
      {
        path.push_back(node_ptr->x);
        node_ptr = node_ptr->parent;
      }
      path.push_back(goal_node_->x);
    }

    Eigen::Vector3d steer(const Eigen::Vector3d &nearest_node_p, const Eigen::Vector3d &rand_node_p, double len)
    {
      Eigen::Vector3d diff_vec = rand_node_p - nearest_node_p;
      double dist = diff_vec.norm();
      if (diff_vec.norm() <= len)
        return rand_node_p;
      else
        return nearest_node_p + diff_vec * len / dist;
    }

    bool greedySteer(const Eigen::Vector3d &x_near, const Eigen::Vector3d &x_target, vector<Eigen::Vector3d> &x_connects, const double len)
    {
      double vec_length = (x_target - x_near).norm();
      Eigen::Vector3d vec_unit = (x_target - x_near) / vec_length;
      x_connects.clear();

      if (vec_length < len)
        return map_ptr_->isSegmentValid(x_near, x_target);

      Eigen::Vector3d x_new, x_pre = x_near;
      double steered_dist = 0;

      while (steered_dist + len < vec_length)
      {
        x_new = x_pre + len * vec_unit;
        if ((!map_ptr_->isStateValid(x_new)) || (!map_ptr_->isSegmentValid(x_new, x_pre)))
          return false;

        x_pre = x_new;
        x_connects.push_back(x_new);
        steered_dist += len;
      }
      return map_ptr_->isSegmentValid(x_target, x_pre);
    }
    double computeH(const Eigen::Vector3d &si, const Eigen::Vector3d &gi)
    {
      Eigen::Vector3d si_gi, si_G, gi_S;
      double si_gi_dist, si_G_dist, gi_S_dist, h;
      si_gi = si - gi;
      si_G = si - goal_node_->x;
      gi_S = gi - start_node_->x;
      si_gi_dist = si_gi.norm();
      si_G_dist = si_G.norm();
      gi_S_dist = gi_S.norm();
      h = brrt_optimize_alpha_ * si_gi_dist + brrt_optimize_beta_ * si_G_dist + brrt_optimize_gamma_ * gi_S_dist;
      return h;
    }
    

    void update_cache_nearest_heuristic(RRTNode3DPtr nodeSi,kdtree *treeA, kdtree *treeB)
    {

      // Iterate through all nodes in treeA

      // Find the nearest node in treeB to the current node in treeA
      // struct kdres *nodesB = kd_nearest_range3(treeB, nodeSi->x[0], nodeSi->x[1], nodeSi->x[2], DBL_MAX);
      struct kdres *nodesB = kd_nearest_n(treeB, nodeSi->x.data(), 30);
      // std::cout << "size of nodesB: " << kd_res_size(nodesB) << std::endl;
      while (!kd_res_end(nodesB))
      {
        RRTNode3DPtr nodeGi = (RRTNode3DPtr)kd_res_item_data(nodesB);
        double h = computeH(nodeSi->x, nodeGi->x);
        cache.insert(nodeSi, treeA, nodeGi, treeB, h);  // same as insert(nodeB, treeB_ptr, nodeA, treeA_ptr, 1.23)
        kd_res_next(nodesB);
      }
      kd_res_free(nodesB);
    }
    Eigen::Vector3d get_sample_valid()
    {
      Eigen::Vector3d x_rand;
      sampler_.samplingOnce(x_rand);
      // samplingOnce(x_rand);
      while (!map_ptr_->isStateValid(x_rand))
      {
        sampler_.samplingOnce(x_rand);
      }
      return x_rand;
    }

    bool intersectRaySphere(const Eigen::Vector3d &A, const Eigen::Vector3d &D, const Eigen::Vector3d &B, double radius, Eigen::Vector3d &intersection, float escape = 0.002)
    {
      Eigen::Vector3d m = A - B;
      double a = D.dot(D), b = 2.0 * D.dot(m), c = m.dot(m) - radius * radius;
      double discriminant = b * b - 4 * a * c;
      if (discriminant < 0)
        return false;

      double sqrt_disc = std::sqrt(discriminant), t1 = (-b - sqrt_disc) / (2 * a), t2 = (-b + sqrt_disc) / (2 * a);
      double t = (std::abs(t1) > escape) ? t1 : ((std::abs(t2) > escape) ? t2 : std::numeric_limits<double>::max());
      if (t == std::numeric_limits<double>::max())
        return false;
      intersection = A + t * D;
      return true;
    }

    Eigen::Vector3d randomPointInCircle(const Eigen::Vector3d& A, const Eigen::Vector3d& B) {
      // Step 1: Get the normal vector of the circle plane
      // std::cout << "[BRRT_Optimize_Case3] randomPointInCircle: A: " << A.transpose() << " B: " << B.transpose() << std::endl;
      Eigen::Vector3d normal = (B - A).normalized();
      double radius = (B - A).norm();
      #ifdef DEBUG
      if (vis_ptr_)
      {
        vis_ptr_->visualize_a_ball(B, radius, "/brrt_optimize/guide", visualization::Color::yellow, 0.3);
      }
#endif
      // Step 2: Create an orthonormal basis (u, v) on the plane
      Eigen::Vector3d u;
      if (std::abs(normal.x()) < 1e-6 && std::abs(normal.y()) < 1e-6) {
          u = Eigen::Vector3d(0, 1, 0).cross(normal).normalized(); // handle edge case
      } else {
          u = Eigen::Vector3d(0, 0, 1).cross(normal).normalized();
      }
      Eigen::Vector3d v = normal.cross(u);
  
      // Step 3: Generate random polar coordinates (r, theta)
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<> dist_angle(0, 2 * M_PI);
      std::uniform_real_distribution<> dist_radius(0, 1);
  
      double theta = dist_angle(gen);
      double r = radius * std::sqrt(dist_radius(gen));  // sqrt for uniformity over area
  
      // Step 4: Compute the point
      Eigen::Vector3d point = A + r * std::cos(theta) * u + r * std::sin(theta) * v;
  
      return point;
  }
    Eigen::Vector3d computeT(const Eigen::Vector3d &A, const Eigen::Vector3d &B, const Eigen::Vector3d &X)
    {
      Eigen::Vector3d AX = X - A;
      Eigen::Vector3d AB = B - A;
      Eigen::Vector3d D = AX + AB;
      double radius = AB.norm();
#ifdef DEBUG
      if (vis_ptr_)
      {
        vis_ptr_->visualize_a_ball(B, radius, "/brrt_optimize/guide", visualization::Color::yellow, 0.3);
      }
#endif
      Eigen::Vector3d intersection;
      if (intersectRaySphere(A, D, B, radius, intersection))
      {
        return intersection;
      }
      else
      {
        return A + D;
      }
    }
#ifdef DEBUG
    void print_vector3d(std::string name, Eigen::Vector3d &p)
    {
      std::cout << name << " x: " << p[0] << " y: " << p[1] << " z: " << p[2] << std::endl;
    }
#endif
    double computePbias(
        double Pinit,
        double h_start_goal,
        const Eigen::Vector3d &sguide,
        const Eigen::Vector3d &tguide)
    {

      if (h_start_goal == 0.0  ||  brrt_optimize_u_p <= 0.00001)
      {
        // Avoid division by zero
        return Pinit;
      }
      double h_sguide_tguide = computeH(sguide, tguide);
      double ratio = brrt_optimize_u_p * (h_start_goal - h_sguide_tguide) / h_start_goal;
      double Pbias = Pinit * std::exp(-ratio);
      return Pbias;
    }
    // Helper to check if a point is inside the Trap Circle (ADDED)
    bool isInsideTrap(const Eigen::Vector3d &p)
    {
        return (p - trap_center_).norm() < trap_radius_;
    }

    // Helper to get neighbors within a radius
    void getNeighbors(kdtree* tree, const Eigen::Vector3d& point, double radius, std::vector<RRTNode3DPtr>& neighbors)
    {
        neighbors.clear();
        struct kdres *res = kd_nearest_range3(tree, point[0], point[1], point[2], radius);
        while (!kd_res_end(res))
        {
            RRTNode3DPtr node = (RRTNode3DPtr)kd_res_item_data(res);
            neighbors.push_back(node);
            kd_res_next(res);
        }
        kd_res_free(res);
    }

    // 1. Choose Best Parent
    // Analyzes neighbors to find the one that offers the cheapest path to x_new
    RRTNode3DPtr chooseBestParent(const std::vector<RRTNode3DPtr>& neighbors, const Eigen::Vector3d& x_new, RRTNode3DPtr nearest_node)
    {
        RRTNode3DPtr best_parent = nearest_node;
        double min_cost = nearest_node->cost_from_start + calDist(nearest_node->x, x_new);

        for (const auto& neighbor : neighbors)
        {
            if (neighbor == nearest_node) continue;

            double dist = calDist(neighbor->x, x_new);
            double potential_cost = neighbor->cost_from_start + dist;

            if (potential_cost < min_cost)
            {
                if (map_ptr_->isSegmentValid(neighbor->x, x_new))
                {
                    min_cost = potential_cost;
                    best_parent = neighbor;
                }
            }
        }
        return best_parent;
    }

    // 2. Rewire (Pruning/Reconnection)
    // Checks if x_new can provide a cheaper path for its neighbors
    void rewire(kdtree* tree, RRTNode3DPtr new_node, const std::vector<RRTNode3DPtr>& neighbors)
    {
        for (const auto& neighbor : neighbors)
        {
            // Do not rewire the parent of the new node (avoids loops)
            if (neighbor == new_node->parent) continue;

            double dist = calDist(new_node->x, neighbor->x);
            double new_cost_via_node = new_node->cost_from_start + dist;

            // If going through new_node is cheaper than neighbor's current path
            if (new_cost_via_node < neighbor->cost_from_start)
            {
                if (map_ptr_->isSegmentValid(new_node->x, neighbor->x))
                {
                    // "Prune" old edge and "Reconnect" to new_node
                    // changeNodeParent handles the cost propagation to neighbor's descendants
                    changeNodeParent(neighbor, new_node, dist);
                }
            }
        }
    }
    // Helper: Simulate Lidar RayCast (Section 3.2) [cite: 268, 299]
    double getRayDistance(const Eigen::Vector3d& start, double angle_rad, double max_range) {
        double r = 0;
        double step = 0.2; // Resolution
        Eigen::Vector3d direction(cos(angle_rad), sin(angle_rad), 0.0); // Planar scan
        
        while (r < max_range) {
            Eigen::Vector3d p = start + direction * r;
            // Check collision. If invalid, return current distance (hit obstacle)
            if (!map_ptr_->isStateValid(p)) return r;
            r += step;
        }
        return max_range;
    }

    // Helper: Randomly select a node from the existing tree [cite: 297]
    RRTNode3DPtr getRandomTreeNode() {
        if (valid_tree_node_nums_ == 0) return start_node_;
        // Simple random selection from the pool of valid nodes
        int idx = std::rand() % valid_tree_node_nums_;
        return nodes_pool_[idx];
    }
    
    // Helper: Get node closest to goal (for exploitation) [cite: 298]
    RRTNode3DPtr getNearestNodeToGoal(kdtree* tree) {
        // Query KD-tree for node nearest to goal coordinates
        struct kdres *res = kd_nearest3(tree, goal_node_->x[0], goal_node_->x[1], goal_node_->x[2]);
        if (kd_res_end(res)) return start_node_;
        RRTNode3DPtr node = (RRTNode3DPtr)kd_res_item_data(res);
        kd_res_free(res);
        return node;
    }

    // MAIN FUNCTION: Algorithm 1 WeightSample 
    Eigen::Vector3d SpatialWeightSample(kdtree* treeA) {
        // 1. Determine "Center" Node for sampling [cite: 238-245]
        RRTNode3DPtr centerNode;
        double rand_val = (double)rand() / RAND_MAX;
        
        // Epsilon-greedy strategy:
        // If rand < epsilon, explore (pick random node). Else, exploit (pick node near goal).
        if (rand_val < sof_epsilon_) {
             centerNode = getRandomTreeNode();
        } else {
             centerNode = getNearestNodeToGoal(treeA);
        }

        // 2. Simulate Lidar (GetBlockLines) [cite: 299]
        std::vector<double> block_weights;
        double total_weight = 0.0;
        
        for (int i = 0; i < sof_n_blocks_; ++i) {
            // Calculate angle for this sector
            double angle = (2.0 * M_PI * i) / sof_n_blocks_;
            
            // Get ray length (radius of free space)
            double radius = getRayDistance(centerNode->x, angle, sof_lidar_range_);
            
            // Calculate Weight: radius^weightGrade [cite: 272]
            double weight = std::pow(radius, sof_weight_grade_);
            block_weights.push_back(weight);
            total_weight += weight;
        }

        // 3. Select Sector based on Weight (Roulette Wheel Selection) [cite: 259-279]
        double rand_weight = ((double)rand() / RAND_MAX) * total_weight;
        double current_sum = 0.0;
        int selected_block_idx = 0;
        
        for (int i = 0; i < sof_n_blocks_; ++i) {
            current_sum += block_weights[i];
            if (rand_weight <= current_sum) {
                selected_block_idx = i;
                break;
            }
        }

        // 4. Sample Point within Selected Block (Sector) [cite: 283]
        // Sampling range: angle [theta_start, theta_end], radius [0, R_block]
        double theta_start = (2.0 * M_PI * selected_block_idx) / sof_n_blocks_;
        double theta_width = (2.0 * M_PI) / sof_n_blocks_;
        
        // Random angle within sector
        double rand_angle = theta_start + ((double)rand() / RAND_MAX) * theta_width;
        
        // Random radius (uniform area sampling implies sqrt(rand))
        double max_r = std::pow(block_weights[selected_block_idx], 1.0/sof_weight_grade_); // recover radius
        double rand_r = max_r * std::sqrt((double)rand() / RAND_MAX);
        
        // Convert back to Cartesian
        Eigen::Vector3d sample_point;
        sample_point[0] = centerNode->x[0] + rand_r * cos(rand_angle);
        sample_point[1] = centerNode->x[1] + rand_r * sin(rand_angle);
        
        // For Z-axis, we can keep it near the node or random within map bounds
        // Assuming 2.5D planning, we might just use the node's Z or random Z.
        // Let's add a small random Z offset or keep it simple:
        sample_point[2] = centerNode->x[2] + ((double)rand()/RAND_MAX - 0.5) * 1.0; 

        return sample_point;
    }
    // Add this inside the private: section
    Eigen::Vector3d GaussianSample(const Eigen::Vector3d& mean, double std_dev) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::normal_distribution<double> d(0.0, std_dev);

        Eigen::Vector3d sample;
        sample[0] = mean[0] + d(gen);
        sample[1] = mean[1] + d(gen);
        sample[2] = mean[2] + (d(gen) * 0.1); 
        
        return sample;
    }
    bool brrt_optimize(const Eigen::Vector3d &s, const Eigen::Vector3d &g)
    {
      ros::Time rrt_start_time = ros::Time::now();
      bool tree_connected = false;
      bool path_reverse = false;

      double h_start_goal = computeH(start_node_->x, goal_node_->x);

      kdtree *kdtree_1 = kd_create(3);
      kdtree *kdtree_2 = kd_create(3);
      kd_insert3(kdtree_1, start_node_->x[0], start_node_->x[1], start_node_->x[2], start_node_);
      kd_insert3(kdtree_2, goal_node_->x[0], goal_node_->x[1], goal_node_->x[2], goal_node_);
      
      RRTNode3DPtr selected_SI = start_node_, selected_GI = goal_node_;
      kdtree *treeA = kdtree_1;
      kdtree *treeB = kdtree_2;

      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<double> dis(0.0, 1.0);

      // --- Trap Variables ---
      bool in_trap_node = false;
      int trapCount = 0;
      int trap_steps_current = 0;
      double h_past = h_start_goal; 
      double h_tmp = h_start_goal;

    #ifdef DEBUG
      std::cout << "[BRRT_Optimize_Case3] Start sampling..." << std::endl;
    #endif
      cache.insert(start_node_, treeA, goal_node_, treeB, h_start_goal);

      for (number_of_iterations_ = 0; number_of_iterations_ < max_iteration_; ++number_of_iterations_)
      {
        Eigen::Vector3d x_new;
        double current_pbias;

        if (in_trap_node)
        {
            // 1. Remove all nodes in the trap region from heuristic cache
            cache.removeNodesInside(trap_center_, trap_radius_, treeA, treeB);

            // 2. Choose another guide nodes pair using Boltzmann distribution
            if (!cache.getBoltzmannPair(treeA, treeB, selected_SI, selected_GI, h_tmp, 5.0)) {
                selected_SI = start_node_; selected_GI = goal_node_; // Fallback
            }

            current_pbias = computePbias(brrt_optimize_p1_, h_start_goal, selected_SI->x, selected_GI->x);
            current_pbias *= 0.1; 
        }
        else
        {
            cache.getMinByTree(treeA, treeB, selected_SI, selected_GI, h_tmp);

            // 2. Check Stagnation
            double h_improvement = h_past - h_tmp;
            
            if (h_improvement < h_threshold_) {
                trapCount++;
                if (trapCount >= trap_count_limit_) {
                    // Enter Trap Solving Mode
                    in_trap_node = true;
                    trap_steps_current = 0;
                    
                    // --- FIX 1: Correct Midpoint and Cap Radius ---
                    trap_center_ = (selected_SI->x + selected_GI->x) * 0.5; // Corrected math (was 0.25)
                    
                    double dist = (selected_SI->x - selected_GI->x).norm();
                    // Cap the radius! Use 25% of distance OR Max 3.0 meters.
                    trap_radius_ = std::min(dist * 0.25, 3.0); 
                    
                    // Trigger trap logic immediately for this step
                    cache.removeNodesInside(trap_center_, trap_radius_, treeA, treeB);
                    current_pbias = 0.0; 
                } else {
                    current_pbias = computePbias(brrt_optimize_p1_, h_start_goal, selected_SI->x, selected_GI->x);
                }
            } else {
                // Heuristic is improving
                trapCount = 0;
                h_past = h_tmp;
                current_pbias = computePbias(brrt_optimize_p1_, h_start_goal, selected_SI->x, selected_GI->x);
            }
        }
        bool sampling_success = false;
        double random01 = dis(gen);

        if (random01 < current_pbias)
        {          
          // Eigen::Vector3d x_tmp
          // if (in_trap_node) {
          //   continue; // Skip iteration
          //   x_tmp = 
          // }
          // else{
          //   Eigen::Vector3d x_tmp = randomPointInCircle(selected_SI->x, selected_GI->x);
          // }
          Eigen::Vector3d x_tmp = randomPointInCircle(selected_SI->x, selected_GI->x);
          x_new = steer(selected_SI->x, x_tmp, steer_length_);
          sampling_success = true;
        }
        else
        {
          Eigen::Vector3d x_rand;

          // --- FIX 2: Gaussian Wiggle Strategy ---
          // If we are struggling (trapCount is high) but not yet in full trap mode,
          // sample NEAR the stuck point to find the "hole" in the clutter.
          // if (trapCount > 5 && !in_trap_node) {
          //     x_rand = GaussianSample(selected_SI->x, 1.5); // Standard deviation 1.5m
          // } else {
          //     x_rand = SpatialWeightSample(treeA);
          // }
          x_rand = SpatialWeightSample(treeA);
          // --- FIX 3: Re-enable Trap Check (Safe now due to capped radius) ---
          int safety = 0;
          while (isInsideTrap(x_rand) && safety < 100){
              x_rand = SpatialWeightSample(treeA);
              safety++;
          }
          
          struct kdres *p_nearestA = kd_nearest3(treeA, x_rand[0], x_rand[1], x_rand[2]);
          if (p_nearestA) {
              RRTNode3DPtr nearest_nodeA = (RRTNode3DPtr)kd_res_item_data(p_nearestA);
              kd_res_free(p_nearestA);
              selected_SI = nearest_nodeA;
              x_new = steer(nearest_nodeA->x, x_rand, steer_length_);
              sampling_success = true;
          }
        }

        if (!sampling_success) continue;

        if ((!map_ptr_->isStateValid(x_new)) || (!map_ptr_->isSegmentValid(selected_SI->x, x_new)))
        {
            std::swap(treeA, treeB);
            path_reverse = !path_reverse;
            continue;
        }

        struct kdres *p_nearestB = kd_nearest3(treeB, x_new[0], x_new[1], x_new[2]);
        if (!p_nearestB) continue;
        RRTNode3DPtr nearest_nodeB = (RRTNode3DPtr)kd_res_item_data(p_nearestB);
        kd_res_free(p_nearestB);
        selected_GI = nearest_nodeB;

        if (valid_tree_node_nums_ + 1 >= max_tree_node_nums_) break;

        // 1. Get Neighbors
        std::vector<RRTNode3DPtr> neighbors;
        getNeighbors(treeA, x_new, rewire_radius_, neighbors);

        // 2. Choose Best Parent (Optimization before insertion)
        RRTNode3DPtr best_parent = chooseBestParent(neighbors, x_new, selected_SI);
        
        double cost_from_parent = calDist(best_parent->x, x_new);
        double cost_from_start = best_parent->cost_from_start + cost_from_parent;
        
        RRTNode3DPtr new_nodeA = addTreeNode(best_parent, x_new, cost_from_start, cost_from_parent);
        kd_insert3(treeA, x_new[0], x_new[1], x_new[2], new_nodeA);
        
        // 3. Rewire (Reconnection / Pruning after insertion)
        rewire(treeA, new_nodeA, neighbors);

        update_cache_nearest_heuristic(new_nodeA, treeA, treeB);

        if (in_trap_node) 
        {
            // Estimate heuristic improvement of the NEW node
            double h_new_est = computeH(new_nodeA->x, nearest_nodeB->x);
            
            if ((h_past - h_new_est) > h_threshold_) 
            {
                // Yes -> Exit trap mode
                in_trap_node = false;
                trapCount = 0;
                h_past = h_new_est;
            }
            else 
            {
                trap_steps_current++;
                if (trap_steps_current >= trap_step_limit_) 
                {
                    in_trap_node = false;
                    trapCount = 0;
                    h_past = h_new_est;
                }
            }
        }

        vector<Eigen::Vector3d> x_connects;
        bool isConnected = greedySteer(nearest_nodeB->x, x_new, x_connects, steer_length_);

        if (!x_connects.empty())
        {
          if (valid_tree_node_nums_ + (int)x_connects.size() >= max_tree_node_nums_) break;

          RRTNode3DPtr new_nodeB_conn = nearest_nodeB;
          for (auto x_connect : x_connects)
          {
            new_nodeB_conn = addTreeNode(new_nodeB_conn, x_connect, new_nodeB_conn->cost_from_start + steer_length_, steer_length_);
            kd_insert3(treeB, x_connect[0], x_connect[1], x_connect[2], new_nodeB_conn);
          }
          update_cache_nearest_heuristic(new_nodeB_conn, treeB, treeA);
        }
        #ifdef DEBUG
        // Visualize the tree at every step
        if (vis_ptr_) {
            visualizeWholeTree();
        }
        
        if (animation_delay_ > 0.0) {
            ros::Duration(animation_delay_).sleep();
        }
        #endif
        if (isConnected)
        {
          tree_connected = true;
          double path_cost = new_nodeA->cost_from_start + nearest_nodeB->cost_from_start + calDist(nearest_nodeB->x, new_nodeA->x);
          if (path_cost < cost_best_)
          {
            vector<Eigen::Vector3d> curr_best_path;
            if (path_reverse)
              fillPath(nearest_nodeB, new_nodeA, curr_best_path);
            else
              fillPath(new_nodeA, nearest_nodeB, curr_best_path);
            path_list_.emplace_back(curr_best_path);
            solution_cost_time_pair_list_.emplace_back(path_cost, (ros::Time::now() - rrt_start_time).toSec());
            cost_best_ = path_cost;
          }
    #ifdef DEBUG
          std::cout << "[BRRT_Optimize_Case3]**********Find path after " << number_of_iterations_ << " iterations" << std::endl;
    #endif
          break;
        }
        else
        {
          std::swap(treeA, treeB);
          path_reverse = !path_reverse;
        }

      } // End of sampling iteration
    #ifdef DEBUG
      visualizeWholeTree();
    #endif
      final_path_use_time_ = (ros::Time::now() - rrt_start_time).toSec();
      if (tree_connected)
      {

    #ifdef DEBUG
        ROS_INFO_STREAM("[BRRT_Optimize_Case3]: find_path_use_time: " << solution_cost_time_pair_list_.front().second << ", length: " << solution_cost_time_pair_list_.front().first);
    #endif
        final_path_ = path_list_.back();
      }
    #ifdef DEBUG
      else if (valid_tree_node_nums_ == max_tree_node_nums_)
      {
        ROS_ERROR_STREAM("[BRRT_Optimize_Case3]: NOT CONNECTED TO GOAL after " << max_tree_node_nums_ << " nodes added to rrt-tree");
      }
      else
      {
        ROS_ERROR_STREAM("[BRRT_Optimize_Case3]: NOT CONNECTED TO GOAL after " << (ros::Time::now() - rrt_start_time).toSec() << " seconds");
      }
    #endif
      return tree_connected;
    }

    void visualizeWholeTree()
    {
      // Sample and visualize the resultant tree
      vector<Eigen::Vector3d> vertice;
      vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> edges;
      vertice.clear();
      edges.clear();
      sampleWholeTree(start_node_, vertice, edges);
      sampleWholeTree(goal_node_, vertice, edges);
      std::vector<visualization::BALL> tree_nodes;
      tree_nodes.reserve(vertice.size());
      visualization::BALL node_p;
      node_p.radius = 0.12;
      for (size_t i = 0; i < vertice.size(); ++i)
      {
        node_p.center = vertice[i];
        tree_nodes.push_back(node_p);
      }
      vis_ptr_->visualize_balls(tree_nodes, "case3/tree_vertice", visualization::Color::yellow, 0.5);
      vis_ptr_->visualize_pairline(edges, "case3/tree_edges", visualization::Color::yellow, 0.05);
    }

    void sampleWholeTree(const RRTNode3DPtr &root, vector<Eigen::Vector3d> &vertice, vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &edges)
    {
      if (root == nullptr)
        return;

      // whatever dfs or bfs
      RRTNode3DPtr node = root;
      std::queue<RRTNode3DPtr> Q;
      Q.push(node);
      while (!Q.empty())
      {
        node = Q.front();
        Q.pop();
        for (const auto &leafptr : node->children)
        {
          vertice.push_back(leafptr->x);
          edges.emplace_back(std::make_pair(node->x, leafptr->x));
          Q.push(leafptr);
        }
      }
    }

  public:
    void samplingOnce(Eigen::Vector3d &sample)
    {
      static int i = 0;
      sample = preserved_samples_[i];
      i++;
      i = i % preserved_samples_.size();
    }

    void setPreserveSamples(const vector<Eigen::Vector3d> &samples)
    {
      preserved_samples_ = samples;
    }
    vector<Eigen::Vector3d> preserved_samples_;
  };

} // namespace path_plan
#endif
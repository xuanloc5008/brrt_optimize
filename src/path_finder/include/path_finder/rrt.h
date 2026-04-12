/*
Copyright (C) 2022 Hongkai Ye (kyle_yeh@163.com)
Modified to implement SOF-RRT* based on Yu et al. (2023)
*/
#ifndef RRT_H
#define RRT_H

#include "occ_grid/occ_map.h"
#include "visualization/visualization.hpp"
#include "sampler.h"
#include "node.h"
#include "kdtree.h"

#include <ros/ros.h>
#include <utility>
#include <queue>
#include <cmath>
#include <random>
#include <algorithm>

namespace path_plan
{
  class RRT
  {
  public:
    RRT(){};
    RRT(const ros::NodeHandle &nh, const env::OccMap::Ptr &mapPtr) : nh_(nh), map_ptr_(mapPtr)
    {
      // Các tham số RRT cơ bản
      nh_.param("RRT/steer_length", steer_length_, 2.0);
      nh_.param("RRT/search_radius", search_radius_, 5.0);
      nh_.param("RRT/search_time", search_time_, 0.5);
      nh_.param("RRT/max_tree_node_nums", max_tree_node_nums_, 5000);
      
      ROS_WARN_STREAM("[SOF-RRT*] param: steer_length: " << steer_length_);
      ROS_WARN_STREAM("[SOF-RRT*] param: search_radius: " << search_radius_);
      ROS_WARN_STREAM("[SOF-RRT*] param: search_time: " << search_time_);
      ROS_WARN_STREAM("[SOF-RRT*] param: max_tree_node_nums: " << max_tree_node_nums_);

      sampler_.setSamplingRange(mapPtr->getOrigin(), mapPtr->getMapSize());

      valid_tree_node_nums_ = 0;
      nodes_pool_.resize(max_tree_node_nums_);
      for (int i = 0; i < max_tree_node_nums_; ++i)
      {
        nodes_pool_[i] = new TreeNode;
      }
      
      // Khởi tạo random engine cho SOF-RRT* logic
      std::random_device rd;
      gen_ = std::mt19937(rd());
      rand01_ = std::uniform_real_distribution<double>(0.0, 1.0);
    }
    ~RRT(){};

    bool plan(const Eigen::Vector3d &s, const Eigen::Vector3d &g)
    {
      reset();
      if (!map_ptr_->isStateValid(s))
      {
        ROS_ERROR("[RRT]: Start pos collide or out of bound");
        return false;
      }
      if (!map_ptr_->isStateValid(g))
      {
        ROS_ERROR("[RRT]: Goal pos collide or out of bound");
        return false;
      }
      /* construct start and goal nodes */
      start_node_ = nodes_pool_[1];
      start_node_->x = s;
      start_node_->cost_from_start = 0.0;
      start_node_->parent = nullptr; 
      
      goal_node_ = nodes_pool_[0];
      goal_node_->x = g;
      goal_node_->cost_from_start = DBL_MAX; 
      valid_tree_node_nums_ = 2;             

      ROS_INFO("[SOF-RRT*]: RRT starts planning a path");
      return rrt(s, g);
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
    int get_valid_tree_node_nums() const { return valid_tree_node_nums_; }
    int get_number_of_iteration() const { return 0; }
    double get_final_path_use_time_() const { return final_path_use_time_; }

    void setVisualizer(const std::shared_ptr<visualization::Visualization> &visPtr)
    {
      vis_ptr_ = visPtr;
    };

  private:
    // nodehandle params
    ros::NodeHandle nh_;

    BiasSampler sampler_;

    double steer_length_;
    double search_radius_;
    double search_time_;
    int max_tree_node_nums_;
    int valid_tree_node_nums_;
    double first_path_use_time_;
    double final_path_use_time_;

    std::vector<TreeNode *> nodes_pool_;
    TreeNode *start_node_;
    TreeNode *goal_node_;
    vector<Eigen::Vector3d> final_path_;
    vector<vector<Eigen::Vector3d>> path_list_;
    vector<std::pair<double, double>> solution_cost_time_pair_list_;

    // environment
    env::OccMap::Ptr map_ptr_;
    std::shared_ptr<visualization::Visualization> vis_ptr_;

    // Random generator for SOF logic
    std::mt19937 gen_;
    std::uniform_real_distribution<double> rand01_;

    // SOF-RRT* Parameters (Hardcoded defaults or could be params)
    double epsilon_ = 0.9;            // Exploration factor [cite: 398]
    double epsilon_floor_ = 0.1;      // Min exploration [cite: 405]
    double gamma_ = 0.999;            // Decay rate [cite: 403]
    double p_goal_ = 0.1;             // Goal sample rate [cite: 397]
    double weight_grade_ = 1.0;       // Weight power [cite: 270]
    int n_blocks_ = 16;               // Simulated Lidar sectors [cite: 268]
    double lidar_max_range_ = 15.0;   // Simulated Lidar range

    void reset()
    {
      final_path_.clear();
      path_list_.clear();
      solution_cost_time_pair_list_.clear();
      for (int i = 0; i < max_tree_node_nums_; ++i) // Reset all nodes
      {
        nodes_pool_[i]->parent = nullptr;
        nodes_pool_[i]->children.clear();
        nodes_pool_[i]->cost_from_start = 0.0;
        nodes_pool_[i]->cost_from_parent = 0.0;
      }
      valid_tree_node_nums_ = 0;
      epsilon_ = 0.9; // Reset exploration factor
    }

    double calDist(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2)
    {
      return (p1 - p2).norm();
    }

    // --- Helper: Sigmoid Function [cite: 367] ---
    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    // --- Helper: Simulated Lidar Raycast [cite: 268] ---
    // Trả về khoảng cách đến vật cản gần nhất theo hướng angle
    double rayCast(const Eigen::Vector3d &start, double angle) {
        Eigen::Vector3d direction(cos(angle), sin(angle), 0.0);
        double dist = 0.0;
        double step = map_ptr_->getResolution(); 
        Eigen::Vector3d current = start;
        
        while (dist < lidar_max_range_) {
            current = start + direction * dist;
            if (!map_ptr_->isStateValid(current)) {
                return dist;
            }
            dist += step;
        }
        return lidar_max_range_;
    }

    // --- Helper: WeightSample Strategy (Algorithm 1) [cite: 227] ---
    Eigen::Vector3d WeightSample() {
        // Update epsilon [cite: 237]
        epsilon_ = std::max(epsilon_ * gamma_, epsilon_floor_);

        // 1. Goal Bias [cite: 238]
        if (rand01_(gen_) < p_goal_) {
            return goal_node_->x;
        }

        // 2. Select Node for Expansion [cite: 240-245]
        TreeNode* chosen_node = nullptr;
        if (rand01_(gen_) < epsilon_) {
             // Random existing node
             int idx = std::rand() % (valid_tree_node_nums_ > 2 ? valid_tree_node_nums_ : 2);
             if (idx == 0) idx = 1; // Skip goal placeholder
             chosen_node = nodes_pool_[idx];
        } else {
             // Nearest to goal (Simplified: linear search for demo)
             double min_dist = DBL_MAX;
             for(int i=1; i<valid_tree_node_nums_; ++i) {
                 double d = calDist(nodes_pool_[i]->x, goal_node_->x);
                 if(d < min_dist) {
                     min_dist = d;
                     chosen_node = nodes_pool_[i];
                 }
             }
             if(!chosen_node) chosen_node = start_node_;
        }

        // 3. Simulated Lidar & Sector Weighting [cite: 246-264]
        struct Sector { double min_a, max_a, r, w; };
        std::vector<Sector> blocks;
        double angle_step = 2.0 * M_PI / n_blocks_;
        double sum_weight = 0.0;

        for (int i = 0; i < n_blocks_; ++i) {
            Sector s;
            s.min_a = i * angle_step;
            s.max_a = (i + 1) * angle_step;
            double center_angle = s.min_a + angle_step / 2.0;
            s.r = rayCast(chosen_node->x, center_angle);
            s.w = std::pow(s.r, weight_grade_); // Weight calculation Eq(6) [cite: 271]
            sum_weight += s.w;
            blocks.push_back(s);
        }

        // 4. Roulette Wheel Selection [cite: 16] (Line 16 in Alg 1)
        double rand_val = rand01_(gen_) * sum_weight;
        double cur_sum = 0.0;
        Sector selected = blocks.back();
        for (const auto& b : blocks) {
            cur_sum += b.w;
            if (rand_val <= cur_sum) {
                selected = b;
                break;
            }
        }

        // 5. Sample within selected block [cite: 283]
        double r = std::sqrt(rand01_(gen_)) * selected.r; // Uniform sampling in circle sector
        double theta = selected.min_a + rand01_(gen_) * (selected.max_a - selected.min_a);
        
        return Eigen::Vector3d(
            chosen_node->x[0] + r * cos(theta),
            chosen_node->x[1] + r * sin(theta),
            chosen_node->x[2] // Assuming 2D/Flat 3D
        );
    }

    // --- Helper: AFBGSteer (Target & Obstacle Tangential Bias) [cite: 338] ---
    // Thay thế hàm steer() gốc bằng logic mới
    Eigen::Vector3d steer(const Eigen::Vector3d &nearest_node_p, const Eigen::Vector3d &rand_node_p, double len)
    {
        // Vector mở rộng cơ bản
        Eigen::Vector3d v_expand = (rand_node_p - nearest_node_p).normalized();

        // Tìm khoảng cách tới vật cản gần nhất (Simulated scan) [cite: 343]
        double min_obs_dist = DBL_MAX;
        Eigen::Vector3d obs_vec(0,0,0);
        for(int i=0; i<8; ++i) { // Quét 8 hướng quanh node
            double ang = i * M_PI / 4.0;
            double d = rayCast(nearest_node_p, ang);
            if(d < min_obs_dist) {
                min_obs_dist = d;
                obs_vec = Eigen::Vector3d(cos(ang), sin(ang), 0) * d; 
            }
        }

        // Tính Goal Bias (Phi) Eq (9) [cite: 360]
        double dist_to_goal = calDist(nearest_node_p, goal_node_->x);
        double max_dist = map_ptr_->getMapSize()(0);
        // Normalize range [-5, 5] cho sigmoid như mô tả
        double phi = len * sigmoid((dist_to_goal / max_dist) * 5.0 - 2.5); 

        // Tính Obstacle Tangential Bias (Eta) Eq (10) [cite: 360]
        double eta = 0.0;
        Eigen::Vector3d v_tangent(0,0,0);
        if (min_obs_dist < 2.0 * len) { // Vùng ảnh hưởng vật cản [cite: 339]
             eta = len * sigmoid((min_obs_dist / (2.0*len)) * 5.0 - 2.5);
             
             // Tính vector tiếp tuyến: xoay v_expand 90 độ
             Eigen::Vector3d t1(-v_expand[1], v_expand[0], 0);
             Eigen::Vector3d t2(v_expand[1], -v_expand[0], 0);
             // Chọn hướng đẩy ra xa vật cản
             Eigen::Vector3d v_obs_dir = obs_vec.normalized();
             if (t1.dot(v_obs_dir) < t2.dot(v_obs_dir)) v_tangent = t1;
             else v_tangent = t2;
        }

        // Tổng hợp vector: Eq (8) [cite: 338]
        Eigen::Vector3d v_goal = (goal_node_->x - nearest_node_p).normalized();
        Eigen::Vector3d total_vec = v_expand + phi * v_goal + eta * v_tangent;
        
        // Trả về điểm mới
        double dist = (rand_node_p - nearest_node_p).norm();
        if (dist <= len && eta < 0.01) return rand_node_p; // Nếu gần và không có vật cản
        
        return nearest_node_p + total_vec.normalized() * len;
    }

    // --- Standard RRT Helpers (Retained) ---
    RRTNode3DPtr addTreeNode(RRTNode3DPtr &parent, const Eigen::Vector3d &state,
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

    void changeNodeParent(RRTNode3DPtr &node, RRTNode3DPtr &parent, const double &cost_from_parent)
    {
      if (node->parent)
        node->parent->children.remove(node); 
      node->parent = parent;
      node->cost_from_parent = cost_from_parent;
      node->cost_from_start = parent->cost_from_start + cost_from_parent;
      parent->children.push_back(node);

      // Cập nhật chi phí cho tất cả con cháu (Propagate cost)
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

    void fillPath(const RRTNode3DPtr &n, vector<Eigen::Vector3d> &path)
    {
      path.clear();
      RRTNode3DPtr node_ptr = n;
      while (node_ptr->parent)
      {
        path.push_back(node_ptr->x);
        node_ptr = node_ptr->parent;
      }
      path.push_back(start_node_->x);
      std::reverse(std::begin(path), std::end(path));
    }

    // --- Main RRT* Logic [cite: 392] ---
    bool rrt(const Eigen::Vector3d &s, const Eigen::Vector3d &g)
    {
      ros::Time rrt_start_time = ros::Time::now();
      bool goal_found = false;

      /* kd tree init */
      kdtree *kd_tree = kd_create(3);
      kd_insert3(kd_tree, start_node_->x[0], start_node_->x[1], start_node_->x[2], start_node_);

      /* main loop */
      int idx = 0;
      for (idx = 0; (ros::Time::now() - rrt_start_time).toSec() < search_time_ && valid_tree_node_nums_ < max_tree_node_nums_; ++idx)
      {
        /* 1. Spatial Probability Weight Sampling [cite: 203] */
        Eigen::Vector3d x_rand = WeightSample(); 

        if (!map_ptr_->isStateValid(x_rand)) continue;

        /* 2. Nearest Node */
        struct kdres *p_nearest = kd_nearest3(kd_tree, x_rand[0], x_rand[1], x_rand[2]);
        if (p_nearest == nullptr) continue;
        RRTNode3DPtr nearest_node = (RRTNode3DPtr)kd_res_item_data(p_nearest);
        kd_res_free(p_nearest);

        /* 3. AFBGSteer (Target & Obstacle Tangential Bias) [cite: 336] */
        Eigen::Vector3d x_new = steer(nearest_node->x, x_rand, steer_length_);

        if (!map_ptr_->isSegmentValid(nearest_node->x, x_new)) continue;

        /* 4. Adaptive R_near [cite: 384] & Find Neighbors (RRT* Logic) */
        double card_v = (double)valid_tree_node_nums_;
        double r_near = std::min(search_radius_, search_radius_ * std::pow(log(card_v)/card_v, 1.0/3.0) * 8.0); // *8 tuning
        
        struct kdres *p_near = kd_nearest_range3(kd_tree, x_new[0], x_new[1], x_new[2], r_near);
        std::vector<TreeNode*> neighbors;
        while (!kd_res_end(p_near)) {
            TreeNode *nb = (TreeNode *)kd_res_item_data(p_near);
            neighbors.push_back(nb);
            kd_res_next(p_near);
        }
        kd_res_free(p_near);

        /* 5. Choose Parent (RRT* Logic) [cite: 387] */
        TreeNode* min_node = nearest_node;
        double min_cost = nearest_node->cost_from_start + calDist(nearest_node->x, x_new);
        
        for(auto* nb : neighbors) {
            double dist = calDist(nb->x, x_new);
            double cost = nb->cost_from_start + dist;
            if(cost < min_cost) {
                if(map_ptr_->isSegmentValid(nb->x, x_new)) {
                    min_cost = cost;
                    min_node = nb;
                }
            }
        }

        /* 6. Add Node */
        RRTNode3DPtr new_node = addTreeNode(min_node, x_new, min_cost, calDist(min_node->x, x_new));
        kd_insert3(kd_tree, x_new[0], x_new[1], x_new[2], new_node);

        /* 7. Rewire (RRT* Logic) [cite: 387] */
        for(auto* nb : neighbors) {
            if(nb == min_node) continue;
            double dist = calDist(new_node->x, nb->x);
            double new_cost = new_node->cost_from_start + dist;
            if(new_cost < nb->cost_from_start) {
                if(map_ptr_->isSegmentValid(new_node->x, nb->x)) {
                    changeNodeParent(nb, new_node, dist);
                }
            }
        }

        /* 8. Check connection to goal */
        double dist_to_goal = calDist(x_new, goal_node_->x);
        if (dist_to_goal <= search_radius_)
        {
          if (map_ptr_->isSegmentValid(x_new, goal_node_->x))
          {
            // Update goal parent if better path found (RRT* convergence)
            double potential_cost = new_node->cost_from_start + dist_to_goal;
            if (potential_cost < goal_node_->cost_from_start)
            {
                if (!goal_found) first_path_use_time_ = (ros::Time::now() - rrt_start_time).toSec();
                
                goal_found = true;
                changeNodeParent(goal_node_, new_node, dist_to_goal);
                
                vector<Eigen::Vector3d> curr_best_path;
                fillPath(goal_node_, curr_best_path);
                path_list_.emplace_back(curr_best_path);
                solution_cost_time_pair_list_.emplace_back(goal_node_->cost_from_start, (ros::Time::now() - rrt_start_time).toSec());
            }
          }
        }
      }

      /* visualization */
      vector<Eigen::Vector3d> vertice;
      vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> edges;
      sampleWholeTree(start_node_, vertice, edges);
      std::vector<visualization::BALL> balls;
      balls.reserve(vertice.size());
      visualization::BALL node_p;
      node_p.radius = 0.2;
      for (size_t i = 0; i < vertice.size(); ++i)
      {
        node_p.center = vertice[i];
        balls.push_back(node_p);
      }
      vis_ptr_->visualize_balls(balls, "tree_vertice", visualization::Color::blue, 1.0);
      vis_ptr_->visualize_pairline(edges, "tree_edges", visualization::Color::yellow, 0.1);

      if (goal_found)
      {
        final_path_use_time_ = (ros::Time::now() - rrt_start_time).toSec();
        fillPath(goal_node_, final_path_);
        ROS_INFO_STREAM("[SOF-RRT*]: found path cost: " << goal_node_->cost_from_start << ", time: " << final_path_use_time_);
      }
      else
      {
        ROS_ERROR_STREAM("[SOF-RRT*]: NOT CONNECTED TO GOAL");
      }
      return goal_found;
    }

    void sampleWholeTree(const RRTNode3DPtr &root, vector<Eigen::Vector3d> &vertice, vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &edges)
    {
      if (root == nullptr) return;
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
      // Not used in SOF-RRT*, replaced by internal WeightSample
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
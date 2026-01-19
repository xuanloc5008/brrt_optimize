/*
Copyright (C) 2022 Hongkai Ye (kyle_yeh@163.com), Longji Yin (ljyin6038@163.com)
Modified: SOF-BRRT (Removed RRT* Rewire Logic)
*/
#ifndef BRRT_H
#define BRRT_H

#include "occ_grid/occ_map.h"
#include "visualization/visualization.hpp"
#include "sampler.h"
#include "node.h"
#include "kdtree.h"

#include <ros/ros.h>
#include <utility>
#include <queue>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
#include <functional>

namespace path_plan
{
  class BRRT
  {
  public:
    BRRT(){};
    BRRT(const ros::NodeHandle &nh, const env::OccMap::Ptr &mapPtr) : nh_(nh), map_ptr_(mapPtr)
    {
      // Các tham số BRRT gốc
      nh_.param("BRRT/steer_length", steer_length_, 2.0);
      nh_.param("BRRT/search_time", search_time_, 1.0);
      nh_.param("BRRT/max_tree_node_nums", max_tree_node_nums_, 5000);
      nh_.param("BRRT/max_iteration", max_iteration_, 10000);
      
      // Các tham số SOF (Weight Sampling & Bias)
      nh_.param("BRRT/epsilon", epsilon_, 1.0);           
      nh_.param("BRRT/epsilon_floor", epsilon_floor_, 0.2);
      nh_.param("BRRT/gamma", gamma_, 0.998);             
      nh_.param("BRRT/weight_grade", weight_grade_, 1.5); 
      nh_.param("BRRT/n_blocks", n_blocks_, 8);          
      nh_.param("BRRT/lidar_radius", lidar_radius_, 60.0);
      
      // Đã bỏ param search_radius vì không còn dùng Rewire

      ROS_WARN_STREAM("[BRRT-SOF] param: steer_length: " << steer_length_);
      ROS_WARN_STREAM("[BRRT-SOF] param: search_time: " << search_time_);

      sampler_.setSamplingRange(mapPtr->getOrigin(), mapPtr->getMapSize());

      valid_tree_node_nums_ = 0;
      nodes_pool_.resize(max_tree_node_nums_);
      for (int i = 0; i < max_tree_node_nums_; ++i)
      {
        nodes_pool_[i] = new TreeNode;
      }
      
      // Init Random
      std::random_device rd;
      gen_ = std::mt19937(rd());
      rand01_ = std::uniform_real_distribution<double>(0.0, 1.0);
    }
    ~BRRT(){};

    bool plan(const Eigen::Vector3d &s, const Eigen::Vector3d &g)
    {
      reset();
      /* construct start and goal nodes */
      start_node_ = nodes_pool_[1];
      start_node_->x = s;
      start_node_->cost_from_start = 0.0;
      start_node_->parent = nullptr;

      goal_node_ = nodes_pool_[0];
      goal_node_->x = g;
      goal_node_->cost_from_start = 0.0;
      goal_node_->parent = nullptr;
      
      valid_tree_node_nums_ = 2; // put start and goal in tree

      vis_ptr_->visualize_a_ball(s, 0.3, "start", visualization::Color::pink);
      vis_ptr_->visualize_a_ball(g, 0.3, "goal", visualization::Color::steelblue);

      return brrt(s, g);
    }

    vector<Eigen::Vector3d> getPath() { return final_path_; }
    vector<vector<Eigen::Vector3d>> getAllPaths() { return path_list_; }
    vector<std::pair<double, double>> getSolutions() { return solution_cost_time_pair_list_; }
    void setVisualizer(const std::shared_ptr<visualization::Visualization> &visPtr) { vis_ptr_ = visPtr; };
    int get_number_of_iteration() { return number_of_iterations_; }
    int get_valid_tree_node_nums() { return valid_tree_node_nums_; }
    void set_test_param(double steer_length) { /* steer_length_ = steer_length; */ }
    double get_final_path_use_time_() { return final_path_use_time_; }

  private:
    ros::NodeHandle nh_;
    BiasSampler sampler_;

    // Parameters
    double steer_length_;
    double search_time_;
    int max_iteration_;
    int max_tree_node_nums_;
    int valid_tree_node_nums_;
    int number_of_iterations_;
    double final_path_use_time_;
    
    // SOF specific
    double epsilon_, epsilon_floor_, gamma_, weight_grade_, lidar_radius_;
    int n_blocks_;
    std::mt19937 gen_;
    std::uniform_real_distribution<double> rand01_;

    double cost_best_;
    std::vector<TreeNode *> nodes_pool_;
    TreeNode *start_node_;
    TreeNode *goal_node_;
    vector<Eigen::Vector3d> final_path_;
    vector<vector<Eigen::Vector3d>> path_list_;
    vector<std::pair<double, double>> solution_cost_time_pair_list_;

    env::OccMap::Ptr map_ptr_;
    std::shared_ptr<visualization::Visualization> vis_ptr_;

    void reset()
    {
      final_path_.clear();
      path_list_.clear();
      cost_best_ = DBL_MAX;

      solution_cost_time_pair_list_.clear();
      for (int i = 0; i < max_tree_node_nums_; i++)
      {
        nodes_pool_[i]->parent = nullptr;
        nodes_pool_[i]->children.clear();
        nodes_pool_[i]->cost_from_start = 0.0;
        nodes_pool_[i]->cost_from_parent = 0.0;
      }
      valid_tree_node_nums_ = 0;
      epsilon_ = 0.9; // Reset exploration rate
    }

    double calDist(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2)
    {
      return (p1 - p2).norm();
    }
    
    // Helper: Sigmoid for AFBGSteer
    double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

    // Helper: RayCast for SOF Weighting
    double rayCast(const Eigen::Vector3d &start, double angle) {
        Eigen::Vector3d dir(cos(angle), sin(angle), 0.0);
        double dist = 0.0;
        double step = map_ptr_->getResolution();
        Eigen::Vector3d current = start;
        while (dist < lidar_radius_) {
            current = start + dir * dist;
            if (!map_ptr_->isStateValid(current)) return dist;
            dist += step;
        }
        return lidar_radius_;
    }

    // [FIXED] Helper: Weight Sample (SOF Logic)
    // Thêm tham số 'tree' để tìm NearestNodeOfGoal
    Eigen::Vector3d weightSample(TreeNode* root_node, const Eigen::Vector3d& target_point, kdtree* tree) {
        epsilon_ = std::max(epsilon_ * gamma_, epsilon_floor_);
        
        // 1. Goal Bias (Dòng 7, 25-27 Algo)
        // Dùng tham số 0.1 hoặc biến p_goal_ nếu có
        if (rand01_(gen_) < 0.2) return target_point;

        TreeNode* chosen_node = nullptr;

        // 2. Chọn Node để mở rộng (Dòng 8-12 Algo)
        // Nếu ngẫu nhiên < epsilon -> Explore (Chọn ngẫu nhiên)
        if (valid_tree_node_nums_ > 2 && rand01_(gen_) < epsilon_) {
             int idx = std::rand() % valid_tree_node_nums_;
             chosen_node = nodes_pool_[idx];
        } 
        // Ngược lại -> Exploit (Chọn node gần đích nhất) - PHẦN BỔ SUNG
        else {
             // Tìm node trong cây hiện tại gần target_point nhất
             struct kdres *p_nearest = kd_nearest3(tree, target_point[0], target_point[1], target_point[2]);
             if (p_nearest) {
                 chosen_node = (TreeNode*)kd_res_item_data(p_nearest);
                 kd_res_free(p_nearest);
             } else {
                 chosen_node = root_node; // Fallback nếu cây rỗng (hiếm khi xảy ra)
             }
        }

        // Lidar simulation & Sector Weighting (Dòng 13-15 Algo)
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
            s.w = std::pow(s.r, weight_grade_);
            sum_weight += s.w;
            blocks.push_back(s);
        }

        // Roulette Wheel Selection (Dòng 16-23 Algo)
        double rand_val = rand01_(gen_) * sum_weight;
        double cur_sum = 0.0;
        Sector selected = blocks.back();
        for (const auto& b : blocks) {
            cur_sum += b.w;
            if (rand_val <= cur_sum) { selected = b; break; }
        }

        // Sample (Dòng 24 Algo)
        double r = std::sqrt(rand01_(gen_)) * selected.r;
        // double r = rand01_(gen_) * selected.r;
        double theta = selected.min_a + rand01_(gen_) * (selected.max_a - selected.min_a);
        
        return Eigen::Vector3d(chosen_node->x[0] + r*cos(theta), chosen_node->x[1] + r*sin(theta), 0.0);
    }
// Helper: AFBG Steer (SOF Logic)
    Eigen::Vector3d AFBGSteer(const Eigen::Vector3d &x_near, const Eigen::Vector3d &x_rand, const Eigen::Vector3d &x_target)
    {
        Eigen::Vector3d v_expand = (x_rand - x_near).normalized();
        
        // [FIX 1] Thêm dấu chấm phẩy bị thiếu
        using dispair = std::pair<double, Eigen::Vector3d>;

        // [FIX 2] Thêm bộ so sánh tùy chỉnh. 
        // Lý do: std::priority_queue mặc định sẽ so sánh cả pair.second (Eigen::Vector3d), 
        // nhưng Eigen không hỗ trợ toán tử < nên gây lỗi biên dịch.
        // Struct này chỉ bảo queue so sánh khoảng cách (first).
        struct DistCompare {
            bool operator()(const dispair& a, const dispair& b) {
                return a.first > b.first; // Min-heap: số nhỏ hơn được ưu tiên lên đầu
            }
        };

        // [FIX 3] Sử dụng DistCompare thay cho std::greater
        std::priority_queue<dispair, std::vector<dispair>, DistCompare> pqueue;

        // Scan for obstacles locally (8 directions)
        double min_obs_dist = DBL_MAX;
        Eigen::Vector3d obs_vec(0,0,0);
        
        // Logic bên dưới giữ nguyên
        for(int i=0; i<n_blocks_; ++i) {
            double ang = i * M_PI / 4.0;
            double d = rayCast(x_near, ang);
            if(d < lidar_radius_) {
              Eigen::Vector3d vec = Eigen::Vector3d(cos(ang), sin(ang), 0)*d;
              pqueue.push({d, vec});
            }
        }

        if (!pqueue.empty()) {
            min_obs_dist = pqueue.top().first;
            obs_vec = pqueue.top().second;
        }

        Eigen::Vector3d total_vec(0,0,0);

        double dist_to_target = calDist(x_near, x_target);
        double max_dist = map_ptr_->getMapSize()(0);
        
        double phi = steer_length_ * sigmoid((dist_to_target / max_dist) * 5.0); 
        Eigen::Vector3d v_target = (x_target - x_near).normalized();

        double eta = 0.0;
        Eigen::Vector3d v_tangent(0,0,0);

        if (min_obs_dist < 2.0 * steer_length_) {
             eta = steer_length_ * sigmoid((min_obs_dist / (2.0 * steer_length_)) * 5.0);
             
             Eigen::Vector3d t1(-v_expand[1], v_expand[0], 0);
             Eigen::Vector3d t2(v_expand[1], -v_expand[0], 0);
             
             Eigen::Vector3d v_obs_dir = obs_vec.normalized(); 
             
             v_tangent = (t1.dot(v_obs_dir) < t2.dot(v_obs_dir)) ? t1 : t2;
             
             total_vec = v_expand + phi * v_target + eta * v_tangent;
        } 
        else {
             total_vec = v_expand + phi * v_target;
        }
        
        return x_near + total_vec.normalized();
    }

    RRTNode3DPtr addTreeNode(RRTNode3DPtr &parent, const Eigen::Vector3d &state,
                             const double &cost_from_start, const double &cost_from_parent)
    {
      RRTNode3DPtr new_node_ptr = nodes_pool_[valid_tree_node_nums_];
      valid_tree_node_nums_++;
      new_node_ptr->parent = parent;
      if(parent) parent->children.push_back(new_node_ptr);
      new_node_ptr->x = state;
      new_node_ptr->cost_from_start = cost_from_start;
      new_node_ptr->cost_from_parent = cost_from_parent;
      return new_node_ptr;
    }

    void fillPath(const RRTNode3DPtr &node_A, const RRTNode3DPtr &node_B, vector<Eigen::Vector3d> &path)
    {
      path.clear();
      // Trace Tree A (from connection back to Start)
      RRTNode3DPtr node_ptr = node_A;
      while (node_ptr) {
        path.push_back(node_ptr->x);
        node_ptr = node_ptr->parent;
      }
      std::reverse(path.begin(), path.end());

      // Trace Tree B (from connection back to Goal)
      node_ptr = node_B;
      while (node_ptr) {
        path.push_back(node_ptr->x);
        node_ptr = node_ptr->parent;
      }
    }

    bool brrt(const Eigen::Vector3d &s, const Eigen::Vector3d &g)
    {
      ros::Time rrt_start_time = ros::Time::now();
      bool tree_connected = false;
      bool path_reverse = false;
      
      /* kd tree init */
      kdtree *kdtree_start = kd_create(3);
      kdtree *kdtree_goal = kd_create(3);
      kd_insert3(kdtree_start, start_node_->x[0], start_node_->x[1], start_node_->x[2], start_node_);
      kd_insert3(kdtree_goal, goal_node_->x[0], goal_node_->x[1], goal_node_->x[2], goal_node_);
      
      kdtree *treeCurrent = kdtree_start;
      kdtree *treeOther = kdtree_goal;
      TreeNode *rootCurrent = start_node_;
      TreeNode *rootOther = goal_node_;

      /* main loop */
      number_of_iterations_ = 0;
      for (number_of_iterations_ = 0; number_of_iterations_ < max_iteration_; ++number_of_iterations_)
      {
        if ((ros::Time::now() - rrt_start_time).toSec() > search_time_) break;

        /* 1. SOF Weight Sampling */
        // Sample biased towards the OTHER tree's root
        Eigen::Vector3d x_rand = weightSample(rootCurrent, rootOther->x, treeCurrent);
        /* 2. Nearest Neighbor in Current Tree */
        struct kdres *p_nearest = kd_nearest3(treeCurrent, x_rand[0], x_rand[1], x_rand[2]);
        if (!p_nearest) continue;
        TreeNode *x_near = (TreeNode *)kd_res_item_data(p_nearest);
        kd_res_free(p_nearest);

        /* 3. SOF AFBGSteer */
        // Steer towards random point, biased by OTHER tree root and Obstacles
        Eigen::Vector3d x_new_pos = AFBGSteer(x_near->x, x_rand, rootOther->x);

        // Check Collision
        if (!map_ptr_->isStateValid(x_new_pos) || !map_ptr_->isSegmentValid(x_near->x, x_new_pos)) {
            // Swap trees and retry
            std::swap(treeCurrent, treeOther);
            std::swap(rootCurrent, rootOther);
            path_reverse = !path_reverse;
            continue;
        }

        if (valid_tree_node_nums_ >= max_tree_node_nums_) break;

        /* 4. Add Node (Standard RRT Logic, NO Rewire/Optimization) */
        // Connect directly to x_near
        double dist_new = calDist(x_near->x, x_new_pos);
        double min_cost = x_near->cost_from_start + dist_new;
        
        TreeNode* x_new_node = addTreeNode(x_near, x_new_pos, min_cost, dist_new);
        kd_insert3(treeCurrent, x_new_pos[0], x_new_pos[1], x_new_pos[2], x_new_node);

        /* 5. Connect to Other Tree (Greedy Connection) */
        struct kdres *p_nearest_other = kd_nearest3(treeOther, x_new_pos[0], x_new_pos[1], x_new_pos[2]);
        if (p_nearest_other) {
            TreeNode *x_connect = (TreeNode *)kd_res_item_data(p_nearest_other);
            kd_res_free(p_nearest_other);
            
            double dist_connect = calDist(x_connect->x, x_new_pos);
            // Check direct connection validity
            if (dist_connect < steer_length_ && map_ptr_->isSegmentValid(x_connect->x, x_new_pos)) {
                
                tree_connected = true;
                double total_cost = x_new_node->cost_from_start + x_connect->cost_from_start + dist_connect;
                
                // For Standard BRRT (non-star), we often stop at first solution.
                // But keeping the best cost logic allows for multiple iterations if desired (though convergence is not guaranteed without RRT*)
                if (total_cost < cost_best_) {
                    cost_best_ = total_cost;
                    vector<Eigen::Vector3d> curr_path;
                    
                    if (!path_reverse) // Current=StartTree, Other=GoalTree
                         fillPath(x_new_node, x_connect, curr_path);
                    else               // Current=GoalTree, Other=StartTree
                         fillPath(x_connect, x_new_node, curr_path);
                    
                    path_list_.emplace_back(curr_path);
                    solution_cost_time_pair_list_.emplace_back(total_cost, (ros::Time::now() - rrt_start_time).toSec());
                    
                    #ifdef DEBUG
                    std::cout << "[BRRT-SOF] Path Found! Cost: " << total_cost << std::endl;
                    #endif
                    
                    // Break immediately for standard BRRT behavior
                    break; 
                }
            }
        }

        /* Swap Trees */
        std::swap(treeCurrent, treeOther);
        std::swap(rootCurrent, rootOther);
        path_reverse = !path_reverse;
      } // End Loop

      kd_free(kdtree_start);
      kd_free(kdtree_goal);

      final_path_use_time_ = (ros::Time::now() - rrt_start_time).toSec();
      if (tree_connected)
      {
        ROS_INFO_STREAM("[BRRT-SOF]: found path cost: " << cost_best_ << ", time: " << final_path_use_time_);
        final_path_ = path_list_.back();
        visualizeWholeTree();
      }
      else
      {
        ROS_ERROR_STREAM("[BRRT-SOF]: NOT CONNECTED");
      }
      return tree_connected;
    }

    void visualizeWholeTree(){
        vector<Eigen::Vector3d> vertice;
        vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> edges;
        sampleWholeTree(start_node_, vertice, edges);
        sampleWholeTree(goal_node_, vertice, edges);
        std::vector<visualization::BALL> tree_nodes;
        visualization::BALL node_p;
        node_p.radius = 0.1;
        for (const auto& v : vertice) {
          node_p.center = v;
          tree_nodes.push_back(node_p);
        }
        vis_ptr_->visualize_balls(tree_nodes, "brrt/tree_vertice", visualization::Color::white, 0.5);
        vis_ptr_->visualize_pairline(edges, "brrt/tree_edges", visualization::Color::white, 0.05);
    }

    void sampleWholeTree(const RRTNode3DPtr &root, vector<Eigen::Vector3d> &vertice, vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &edges)
    {
      if (root == nullptr) return;
      RRTNode3DPtr node = root;
      std::queue<RRTNode3DPtr> Q;
      Q.push(node);
      while (!Q.empty()) {
        node = Q.front(); Q.pop();
        for (const auto &leafptr : node->children) {
          vertice.push_back(leafptr->x);
          edges.emplace_back(std::make_pair(node->x, leafptr->x));
          Q.push(leafptr);
        }
      }
    }

  public:
    void samplingOnce(Eigen::Vector3d &sample) {}
    void setPreserveSamples(const vector<Eigen::Vector3d> &samples) { preserved_samples_ = samples; }
    vector<Eigen::Vector3d> preserved_samples_;
  };

} // namespace path_plan
#endif
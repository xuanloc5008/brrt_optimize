/*
Implementation of Bidirectional PBG-RRT and BG-RRT
Based on the paper: "A Heuristic Rapidly-Exploring Random Trees Method for Manipulator Motion Planning"
*/
#ifndef BPBG_RRT_H
#define BPBG_RRT_H

#include "occ_grid/occ_map.h"
#include "visualization/visualization.hpp"
#include "sampler.h"
#include "node.h"
#include "kdtree.h"

#include <ros/ros.h>
#include <utility>
#include <queue>
#include <algorithm>
#include <random>
#include <cmath>

namespace path_plan
{
  class BPBG_RRT
  {
  public:
    BPBG_RRT(){};
    BPBG_RRT(const ros::NodeHandle &nh, const env::OccMap::Ptr &mapPtr, bool use_heuristic_probability = true) 
      : nh_(nh), map_ptr_(mapPtr), use_heuristic_probability_(use_heuristic_probability)
    {
      nh_.param("BPBG_RRT/steer_length", steer_length_, 0.0);
      nh_.param("BPBG_RRT/search_time", search_time_, 0.0);
      nh_.param("BPBG_RRT/max_tree_node_nums", max_tree_node_nums_, 0);
      
      // Algorithm specific parameters
      nh_.param("BPBG_RRT/heuristic_probability_P", heuristic_probability_P_, 0.1);
      nh_.param("BPBG_RRT/bg_weight", bg_weight_, 0.5);
      nh_.param("BPBG_RRT/bg_d_scale", bg_d_scale_, 100.0);

      // Fallback: if params missing or invalid, try the RRT namespace used by launch files
      {
        double tmp_d; int tmp_i;
        if (steer_length_ <= 0.0) {
          if (nh_.getParam("RRT/steer_length", tmp_d) && tmp_d > 0.0) {
            steer_length_ = tmp_d;
          }
        }
        if (search_time_ <= 0.0) {
          if (nh_.getParam("RRT/search_time", tmp_d) && tmp_d > 0.0) {
            search_time_ = tmp_d;
          }
        }
        if (max_tree_node_nums_ <= 2) {
          if (nh_.getParam("RRT/max_tree_node_nums", tmp_i) && tmp_i > 2) {
            max_tree_node_nums_ = tmp_i;
          }
        }

        // Defensive defaults
        if (steer_length_ <= 0.0) { steer_length_ = 1.0; }
        if (search_time_ <= 0.0) { search_time_ = 5.0; }
        if (max_tree_node_nums_ <= 2) { max_tree_node_nums_ = 10000; }
        max_iteration_ = 200000;
      }

      sampler_.setSamplingRange(mapPtr->getOrigin(), mapPtr->getMapSize());

      valid_tree_node_nums_ = 0;
      nodes_pool_.resize(max_tree_node_nums_);
      for (int i = 0; i < max_tree_node_nums_; ++i)
      {
        nodes_pool_[i] = new TreeNode;
      }
      
      // Setup random generator for heuristic probability
      rand_gen_ = std::mt19937(std::random_device{}());
      dist_0_1_ = std::uniform_real_distribution<double>(0.0, 1.0);
    }
    ~BPBG_RRT(){};

    bool plan(const Eigen::Vector3d &s, const Eigen::Vector3d &g)
    {
      reset();
      start_node_ = nodes_pool_[1];
      start_node_->x = s;
      start_node_->cost_from_start = 0.0;
      goal_node_ = nodes_pool_[0];
      goal_node_->x = g;
      goal_node_->cost_from_start = 0.0; 
      valid_tree_node_nums_ = 2;         

      return bpbg_rrt(s, g);
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

    void setVisualizer(const std::shared_ptr<visualization::Visualization> &visPtr)
    {
      vis_ptr_ = visPtr;
    }
    
    int get_number_of_iteration(){
      return number_of_iterations_;
    }
    
    int get_valid_tree_node_nums()
    {
      return valid_tree_node_nums_;
    }
    
    void set_test_param(double steer_length){
      steer_length_ = steer_length;
    }
    
    double get_final_path_use_time_()
    {
      return final_path_use_time_;
    }
    
  private:
    ros::NodeHandle nh_;
    BiasSampler sampler_;

    bool use_heuristic_probability_;
    double heuristic_probability_P_;
    double bg_weight_;
    double bg_d_scale_;
    
    std::mt19937 rand_gen_;
    std::uniform_real_distribution<double> dist_0_1_;

    double steer_length_;
    double search_time_;
    int max_iteration_;
    int max_tree_node_nums_;
    int valid_tree_node_nums_;
    int number_of_iterations_;
    double first_path_use_time_;
    double final_path_use_time_;

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

    Eigen::Vector3d mix_steer(const Eigen::Vector3d &q_near, const Eigen::Vector3d &q_rand, const Eigen::Vector3d &q_target, double len)
    {
      Eigen::Vector3d vec_rand = q_rand - q_near;
      double dist_rand = vec_rand.norm();
      if (dist_rand < 1e-6) return q_near;
      Eigen::Vector3d u_rand = vec_rand / dist_rand;

      Eigen::Vector3d vec_goal = q_target - q_near;
      double dist_goal = vec_goal.norm();
      if (dist_goal < 1e-6) return q_target;
      Eigen::Vector3d u_goal = vec_goal / dist_goal;

      // Calculate bias-goal factor \phi
      double phi = bg_weight_ * len * std::exp(-dist_goal / bg_d_scale_);

      Eigen::Vector3d step_vec = len * u_rand + phi * u_goal;
      
      double step_dist = step_vec.norm();
      if (step_dist > len) {
         step_vec = step_vec * (len / step_dist);
      }

      return q_near + step_vec;
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

      if(vec_length < len)
        return map_ptr_->isSegmentValid(x_near,x_target);

      Eigen::Vector3d x_new, x_pre = x_near;
      double steered_dist = 0;
      
      while(steered_dist + len < vec_length)
      {
        x_new = x_pre + len * vec_unit; 
        if( (!map_ptr_->isStateValid(x_new)) || (!map_ptr_->isSegmentValid(x_new,x_pre)) )
          return false;

        x_pre = x_new; 
        x_connects.push_back(x_new);
        steered_dist += len;
      }
      return map_ptr_->isSegmentValid(x_target,x_pre);
    }

    bool bpbg_rrt(const Eigen::Vector3d &s, const Eigen::Vector3d &g)
    {
      ros::Time rrt_start_time = ros::Time::now();
      bool tree_connected = false;
      bool path_reverse = false;
      
      kdtree *kdtree_1 = kd_create(3);
      kdtree *kdtree_2 = kd_create(3);
      kd_insert3(kdtree_1, start_node_->x[0], start_node_->x[1], start_node_->x[2], start_node_);
      kd_insert3(kdtree_2, goal_node_->x[0], goal_node_->x[1], goal_node_->x[2], goal_node_);
      
      kdtree *treeA = kdtree_1;
      kdtree *treeB = kdtree_2;

      number_of_iterations_ = 0;
      for (number_of_iterations_ = 0; number_of_iterations_ < max_iteration_; ++number_of_iterations_)
      {
        Eigen::Vector3d x_rand;
        Eigen::Vector3d q_target = path_reverse ? start_node_->x : goal_node_->x;
        
        // Heuristic Probability: sample goal occasionally
        bool used_heuristic = false;
        if (use_heuristic_probability_ && dist_0_1_(rand_gen_) <= heuristic_probability_P_) {
            x_rand = q_target;
            used_heuristic = true;
        } else {
            sampler_.samplingOnce(x_rand);
        }

        struct kdres *p_nearestA = kd_nearest3(treeA, x_rand[0], x_rand[1], x_rand[2]);
        if (p_nearestA == nullptr)
        {
          continue;
        }
        RRTNode3DPtr nearest_nodeA = (RRTNode3DPtr)kd_res_item_data(p_nearestA);
        kd_res_free(p_nearestA);

        Eigen::Vector3d x_new;
        if (used_heuristic) {
            x_new = steer(nearest_nodeA->x, x_rand, steer_length_);
        } else {
            x_new = mix_steer(nearest_nodeA->x, x_rand, q_target, steer_length_);
        }
        
        if ( (!map_ptr_->isStateValid(x_new)) || (!map_ptr_->isSegmentValid(nearest_nodeA->x, x_new)) ) 
        {
          std::swap(treeA, treeB);
          path_reverse = !path_reverse;
          continue;
        }

        double dist_from_A = nearest_nodeA->cost_from_start + calDist(nearest_nodeA->x, x_new);
        RRTNode3DPtr new_nodeA(nullptr);
        if (valid_tree_node_nums_ + 1 >= max_tree_node_nums_)
        {
           valid_tree_node_nums_ = max_tree_node_nums_; 
          break;
        }
        new_nodeA = addTreeNode(nearest_nodeA, x_new, dist_from_A, calDist(nearest_nodeA->x, x_new));
        kd_insert3(treeA, x_new[0], x_new[1], x_new[2], new_nodeA);

        struct kdres *p_nearestB = kd_nearest3(treeB, x_new[0], x_new[1], x_new[2]);
        if (p_nearestB == nullptr)
        {
          continue;
        }
        RRTNode3DPtr nearest_nodeB = (RRTNode3DPtr)kd_res_item_data(p_nearestB);
        kd_res_free(p_nearestB);

        vector<Eigen::Vector3d> x_connects;
        bool isConnected = greedySteer(nearest_nodeB->x, x_new, x_connects, steer_length_);
        
        RRTNode3DPtr new_nodeB = nearest_nodeB;
        if(!x_connects.empty()){
          if( valid_tree_node_nums_ + (int)x_connects.size() >= max_tree_node_nums_ ){
            valid_tree_node_nums_ = max_tree_node_nums_; 
            break;
          }

          for(auto x_connect: x_connects){
            new_nodeB = addTreeNode(new_nodeB, x_connect, new_nodeB->cost_from_start + calDist(new_nodeB->x, x_connect), calDist(new_nodeB->x, x_connect));
            kd_insert3(treeB, x_connect[0], x_connect[1], x_connect[2], new_nodeB);
          }
        }
        
        if(isConnected){
          tree_connected = true;
          double path_cost = new_nodeA->cost_from_start + new_nodeB->cost_from_start + calDist(new_nodeB->x, new_nodeA->x);
          if(path_cost < cost_best_)
          { 
            vector<Eigen::Vector3d> curr_best_path;
            if(path_reverse)
              fillPath(new_nodeB, new_nodeA, curr_best_path);
            else
              fillPath(new_nodeA, new_nodeB, curr_best_path);
            path_list_.emplace_back(curr_best_path);
            solution_cost_time_pair_list_.emplace_back(path_cost, (ros::Time::now() - rrt_start_time).toSec());
            cost_best_ = path_cost;
          }
          break;
        }
        std::swap(treeA, treeB);
        path_reverse = !path_reverse;
      }
      
      final_path_use_time_ = (ros::Time::now() - rrt_start_time).toSec();
      if (tree_connected)
      {
        final_path_ = path_list_.back();
      }
      return tree_connected;
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

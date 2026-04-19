/*
Copyright (C) 2022 Hongkai Ye (kyle_yeh@163.com), Longji Yin (ljyin6038@163.com )
*/
#ifndef BRRT_OPTIMIZE_H
#define BRRT_OPTIMIZE_H

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
#include <unistd.h> // For usleep

namespace path_plan
{
  class BRRT_Optimize
  {
  public:
    BRRT_Optimize() {};
    BRRT_Optimize(const ros::NodeHandle &nh, const env::OccMap::Ptr &mapPtr) : nh_(nh), map_ptr_(mapPtr)
    {
      nh_.param("BRRT/steer_length", steer_length_, 0.0);
      nh_.param("BRRT/search_time", search_time_, 0.0);
      nh_.param("BRRT/max_tree_node_nums", max_tree_node_nums_, 0);

      nh_.param("BRRT_Optimize/p1", brrt_optimize_p1_, 0.8);
      nh_.param("BRRT_Optimize/u_p", brrt_optimize_u_p, 2.0);
      nh_.param("BRRT_Optimize/step", brrt_optimize_step_, 1.0);

      nh_.param("BRRT_Optimize/alpha", brrt_optimize_alpha_, 0.5);
      nh_.param("BRRT_Optimize/beta", brrt_optimize_beta_, 0.3);
      nh_.param("BRRT_Optimize/gamma", brrt_optimize_gamma_, 0.5);
      nh_.param("BRRT_Optimize/max_iteration", max_iteration_, 0);
      nh_.param("BRRT_Optimize/enable2d", brrt_enable_2d, true);

      // --- Case 3 Params ---
      nh_.param("BRRT/epsilon", epsilon_, 1.0);
      nh_.param("BRRT/epsilon_floor", epsilon_floor_, 0.2);
      nh_.param("BRRT/gamma", gamma_, 0.998);
      nh_.param("BRRT/weight_grade", weight_grade_, 1.0);
      nh_.param("BRRT/n_blocks", n_blocks_, 8);
      nh_.param("BRRT/lidar_radius", lidar_radius_, 5.0);
      // ---------------------

      ROS_WARN_STREAM("[BRRT_Optimize] param: steer_length: " << steer_length_);
      ROS_WARN_STREAM("[BRRT_Optimize] param: search_time: " << search_time_);
      ROS_WARN_STREAM("[BRRT_Optimize] param: max_tree_node_nums: " << max_tree_node_nums_);

      sampler_.setSamplingRange(mapPtr->getOrigin(), mapPtr->getMapSize());

      valid_tree_node_nums_ = 0;
      nodes_pool_.resize(max_tree_node_nums_);
      for (int i = 0; i < max_tree_node_nums_; ++i)
      {
        nodes_pool_[i] = new TreeNode;
      }

      std::random_device rd;
      gen_ = std::mt19937(rd());
      rand01_ = std::uniform_real_distribution<double>(0.0, 1.0);
    }

    ~BRRT_Optimize()
    {
      for (auto node : nodes_pool_)
      {
        if (node)
          delete node;
      }
      nodes_pool_.clear();
    };

    bool plan(const Eigen::Vector3d &s, const Eigen::Vector3d &g)
    {
      reset();
      /* Construct start and goal nodes */
      start_node_ = nodes_pool_[1];
      start_node_->x = s;
      start_node_->cost_from_start = 0.0;

      goal_node_ = nodes_pool_[0];
      goal_node_->x = g;
      goal_node_->cost_from_start = 0.0;

      valid_tree_node_nums_ = 2;
      cache.clear(); // Important: reset heuristic cache
      return brrt_optimize(s, g);
    }

    vector<Eigen::Vector3d> getPath() { return final_path_; }
    vector<vector<Eigen::Vector3d>> getAllPaths() { return path_list_; }
    vector<std::pair<double, double>> getSolutions() { return solution_cost_time_pair_list_; }

    void setVisualizer(const std::shared_ptr<visualization::Visualization> &visPtr)
    {
      vis_ptr_ = visPtr;
    };
    int get_number_of_iteration() { return number_of_iterations_; }
    int get_valid_tree_node_nums() { return valid_tree_node_nums_; }

  private:
    ros::NodeHandle nh_;
    BiasSampler sampler_;
    double brrt_optimize_p1_, brrt_optimize_u_p, brrt_optimize_step_;
    double brrt_optimize_alpha_, brrt_optimize_beta_, brrt_optimize_gamma_;
    int max_iteration_;
    double steer_length_, search_time_;
    int max_tree_node_nums_, number_of_iterations_, valid_tree_node_nums_;
    double final_path_use_time_;
    bool brrt_enable_2d;

    // --- Case 3 Variables ---
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
    HeuristicCache cache;

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
      epsilon_ = 0.9;
    }

    double calDist(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2)
    {
      return (p1 - p2).norm();
    }

    //---------------------------------CASE 3 METHODS----------------------------------
    double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

    double rayCast(const Eigen::Vector3d &start, double angle)
    {
      Eigen::Vector3d dir(cos(angle), sin(angle), 0.0);
      double dist = 0.0;
      double step = map_ptr_->getResolution();
      if (step <= 0.001)
        step = 0.1;

      Eigen::Vector3d current = start;
      while (dist < lidar_radius_)
      {
        current = start + dir * dist;
        if (!map_ptr_->isStateValid(current))
          return dist;
        dist += step;
      }
      return lidar_radius_;
    }

  private:
    // Helper: tính độ thoáng xung quanh vị trí (min clearance)
    double getLocalClearance(const Eigen::Vector3d &pos, int n_rays = 24)
    {
      double min_r = lidar_radius_;
      double angle_step = 2.0 * M_PI / n_rays;
      for (int i = 0; i < n_rays; ++i)
      {
        double ang = i * angle_step;
        double d = rayCast(pos, ang);
        if (d < min_r)
          min_r = d;
      }
      return min_r;
    }

    // Adaptive stepsize theo không gian xung quanh
    double getAdaptiveSteerLength(const Eigen::Vector3d &pos)
    {
      if (!brrt_enable_2d)
        return steer_length_; // giữ nguyên nếu dùng 3D

      double clearance = getLocalClearance(pos, 24);

      // Vùng thoáng → step lớn (tối đa 1/2 R)
      double max_step = 0.5 * lidar_radius_;
      double adaptive = std::max(steer_length_, std::min(max_step, 0.85 * clearance));

      return adaptive;
    }

    Eigen::Vector3d weightSample(TreeNode *root_node, const Eigen::Vector3d &target_point, kdtree *tree, bool rand_sampling)
    {
      epsilon_ = std::max(epsilon_ * gamma_, epsilon_floor_);

      // if (rand01_(gen_) < 0.3) return target_point;

      TreeNode *chosen_node = nullptr;
      if (rand_sampling == true)
      {
        // Logic for RRT Expansion
        if (valid_tree_node_nums_ > 2 && rand01_(gen_) < epsilon_)
        {
          int idx = std::rand() % valid_tree_node_nums_;
          chosen_node = nodes_pool_[idx];

          // [MODIFIED] Prevent picking Root Node if possible
          int attempts = 0;
          while (chosen_node == root_node && attempts < 100)
          {
            idx = std::rand() % valid_tree_node_nums_;
            chosen_node = nodes_pool_[idx];
            attempts++;
          }
        }
        else
        {
          struct kdres *p_nearest = kd_nearest3(tree, target_point[0], target_point[1], target_point[2]);
          if (p_nearest)
          {
            chosen_node = (TreeNode *)kd_res_item_data(p_nearest);
            kd_res_free(p_nearest);
          }
          else
          {
            chosen_node = root_node;
          }
        }
      }
      else
        chosen_node = root_node;

      if (!chosen_node)
        chosen_node = root_node;

      struct Sector
      {
        double min_a, max_a, r, w;
      };
      std::vector<Sector> blocks;
      double angle_step = 2.0 * M_PI / n_blocks_;
      double sum_weight = 0.0;

      std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> sector_rays;

      for (int i = 0; i < n_blocks_; ++i)
      {
        Sector s;
        s.min_a = i * angle_step;
        s.max_a = (i + 1) * angle_step;
        double center_angle = s.min_a + angle_step / 2.0;
        s.r = rayCast(chosen_node->x, center_angle);
        s.w = std::pow(s.r, weight_grade_);
        sum_weight += s.w;
        blocks.push_back(s);

#ifdef DEBUG
        if (vis_ptr_)
        {
          Eigen::Vector3d ray_end = chosen_node->x + Eigen::Vector3d(cos(center_angle), sin(center_angle), 0) * s.r;
          sector_rays.push_back({chosen_node->x, ray_end});
        }
#endif
      }

#ifdef DEBUG
      if (vis_ptr_)
      {
        if (rand_sampling)
        {
          vis_ptr_->visualize_pairline(sector_rays, "/brrt_optimize/sectors_all", visualization::Color::green, 0.2);
          usleep(500000); // 0.2s: Sector visualization
        }
        vis_ptr_->visualize_pairline(sector_rays, "/brrt_optimize/sectors_all", visualization::Color::steelblue, 0.2);
        usleep(500000); // 0.2s: Sector visualization
      }
#endif

      double rand_val = rand01_(gen_) * sum_weight;
      double cur_sum = 0.0;
      Sector selected = blocks.empty() ? Sector{0, 0, 0, 0} : blocks.back();

      for (const auto &b : blocks)
      {
        cur_sum += b.w;
        if (rand_val <= cur_sum)
        {
          selected = b;
          break;
        }
      }

#ifdef DEBUG
      if (vis_ptr_)
      {
        Eigen::Vector3d p1 = chosen_node->x + Eigen::Vector3d(cos(selected.min_a), sin(selected.min_a), 0) * selected.r;
        Eigen::Vector3d p2 = chosen_node->x + Eigen::Vector3d(cos(selected.max_a), sin(selected.max_a), 0) * selected.r;
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> selected_lines;
        selected_lines.push_back({chosen_node->x, p1});
        selected_lines.push_back({chosen_node->x, p2});
        if (rand_sampling)
        {
          vis_ptr_->visualize_pairline(selected_lines, "/brrt_optimize/sector_selected", visualization::Color::cyan, 0.2);
        }
        vis_ptr_->visualize_pairline(selected_lines, "/brrt_optimize/sector_selected", visualization::Color::pink, 0.2);
      }
#endif
      Eigen::Vector3d final_point;
      double r = rand01_(gen_) * selected.r;
      double theta = selected.min_a + rand01_(gen_) * (selected.max_a - selected.min_a);
      final_point = Eigen::Vector3d(chosen_node->x[0] + r * cos(theta), chosen_node->x[1] + r * sin(theta), 0.0);
      while (!map_ptr_->isStateValid(final_point))
      {
        double r = rand01_(gen_) * selected.r;
        double theta = selected.min_a + rand01_(gen_) * (selected.max_a - selected.min_a);
        final_point = Eigen::Vector3d(chosen_node->x[0] + r * cos(theta), chosen_node->x[1] + r * sin(theta), 0.0);
      }
#ifdef DEBUG
      if (vis_ptr_)
      {
        vis_ptr_->visualize_a_ball(final_point, 0.4, "/brrt_optimize/sampled_point_sector", visualization::Color::white);
         usleep(500000); // 0.2s: Sample point visualization
      }
#endif

      return final_point;
    }

    Eigen::Vector3d AFBGSteer(const Eigen::Vector3d &x_near, const Eigen::Vector3d &x_rand, const Eigen::Vector3d &x_target, double /*steer_length_ ignored*/)
    {
      double adaptive_len = getAdaptiveSteerLength(x_near); // ← adaptive

      // [FIX] Avoid NaN division by zero
      Eigen::Vector3d diff = x_rand - x_near;
      if (diff.norm() < 1e-6)
      {
        return x_near;
      }
      Eigen::Vector3d v_expand = diff.normalized();

      using dispair = std::pair<double, Eigen::Vector3d>;
      struct DistCompare
      {
        bool operator()(const dispair &a, const dispair &b) { return a.first > b.first; }
      };
      std::priority_queue<dispair, std::vector<dispair>, DistCompare> pqueue;

      double min_obs_dist = DBL_MAX;
      Eigen::Vector3d obs_vec(0, 0, 0);
      int n_blocks_ = 32;
      for (int i = 0; i < n_blocks_; ++i)
      {
        double ang = i * (2.0 * M_PI / n_blocks_);
        double d = rayCast(x_near, ang);
        if (d < lidar_radius_)
        {
          Eigen::Vector3d vec = Eigen::Vector3d(cos(ang), sin(ang), 0) * d;
          pqueue.push({d, vec});
        }
      }
      if (!pqueue.empty())
      {
        min_obs_dist = pqueue.top().first;
        obs_vec = pqueue.top().second;
      }

      Eigen::Vector3d total_vec(0, 0, 0);
      double dist_to_target = calDist(x_near, x_target);
      double max_dist = map_ptr_->getMapSize()(0);
      double phi = adaptive_len * sigmoid((dist_to_target / max_dist) * 5.0); // ← adaptive

      Eigen::Vector3d v_target = (x_target - x_near).normalized();
      double eta = 0.0;
      Eigen::Vector3d v_tangent(0, 0, 0);

      if (min_obs_dist < 2.0 * adaptive_len)
      {                                                                            // ← adaptive
        eta = adaptive_len * sigmoid((min_obs_dist / (2.0 * adaptive_len)) * 5.0); // ← adaptive
        Eigen::Vector3d t1(-v_expand[1], v_expand[0], 0);
        Eigen::Vector3d t2(v_expand[1], -v_expand[0], 0);
        Eigen::Vector3d v_obs_dir = (obs_vec.norm() > 1e-6) ? obs_vec.normalized() : Eigen::Vector3d(1, 0, 0);
        v_tangent = (t1.dot(v_obs_dir) < t2.dot(v_obs_dir)) ? t1 : t2;
        total_vec = v_expand + phi * v_target + eta * v_tangent;
      }
      else
      {
        total_vec = v_expand + phi * v_target;
      }

      if (total_vec.norm() > 1e-6)
        return x_near + total_vec.normalized() * adaptive_len; // ← adaptive
      else
        return x_near;
    }

    // --- HEURISTIC CORE METHODS ---
    double computeH(const Eigen::Vector3d &si, const Eigen::Vector3d &gi)
    {
      double si_gi_dist = (si - gi).norm();
      double si_G_dist = (si - goal_node_->x).norm();
      double gi_S_dist = (gi - start_node_->x).norm();
      // Weights alpha, beta, gamma determine the "pull" strength
      return brrt_optimize_alpha_ * si_gi_dist +
             brrt_optimize_beta_ * si_G_dist +
             brrt_optimize_gamma_ * gi_S_dist;
    }

    double computePbias(double Pinit, double h_start_goal, const Eigen::Vector3d &sguide, const Eigen::Vector3d &tguide)
    {
      if (h_start_goal == 0.0 || brrt_optimize_u_p <= 0.00001)
        return Pinit;
      double h_sguide_tguide = computeH(sguide, tguide);
      double ratio = brrt_optimize_u_p * (h_start_goal - h_sguide_tguide) / h_start_goal;
      return Pinit * std::exp(-ratio);
    }

    void update_cache_nearest_heuristic(RRTNode3DPtr nodeSi, kdtree *treeA, kdtree *treeB)
    {

      // Iterate through all nodes in treeA

      // Find the nearest node in treeB to the current node in treeA
      struct kdres *nodesB = kd_nearest_range3(treeB, nodeSi->x[0], nodeSi->x[1], nodeSi->x[2], DBL_MAX);
      while (!kd_res_end(nodesB))
      {
        RRTNode3DPtr nodeGi = (RRTNode3DPtr)kd_res_item_data(nodesB);
        double h = computeH(nodeSi->x, nodeGi->x);
        cache.insert(nodeSi, treeA, nodeGi, treeB, h); // same as insert(nodeB, treeB_ptr, nodeA, treeA_ptr, 1.23)
        kd_res_next(nodesB);
      }
      kd_res_free(nodesB);
    }
    RRTNode3DPtr addTreeNode(RRTNode3DPtr &parent, const Eigen::Vector3d &state,
                             const double &cost_from_start, const double &cost_from_parent)
    {
      if (valid_tree_node_nums_ >= max_tree_node_nums_)
        return nullptr;
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
      // Trace back from Tree A to Start
      RRTNode3DPtr node_ptr = node_A;
      while (node_ptr)
      {
        path.push_back(node_ptr->x);
        node_ptr = node_ptr->parent;
      }
      std::reverse(path.begin(), path.end());

      // Trace back from Tree B to Goal
      node_ptr = node_B;
      while (node_ptr)
      {
        path.push_back(node_ptr->x);
        node_ptr = node_ptr->parent;
      }
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

    Eigen::Vector3d get_sample_valid()
    {
      Eigen::Vector3d x_rand;
      sampler_.samplingOnce(x_rand);
      while (!map_ptr_->isStateValid(x_rand))
      {
        sampler_.samplingOnce(x_rand);
      }
      return x_rand;
    }

    bool brrt_optimize(const Eigen::Vector3d &s, const Eigen::Vector3d &g)
    {
      ros::Time rrt_start_time = ros::Time::now();
      bool tree_connected = false;
      bool path_reverse = false;

      double h_start_goal = computeH(start_node_->x, goal_node_->x);

      // insert start and goal node to cache
      /* kd tree init */
      kdtree *kdtree_1 = kd_create(3);
      kdtree *kdtree_2 = kd_create(3);
      // Add start and goal nodes to kd trees
      kd_insert3(kdtree_1, start_node_->x[0], start_node_->x[1], start_node_->x[2], start_node_);
      kd_insert3(kdtree_2, goal_node_->x[0], goal_node_->x[1], goal_node_->x[2], goal_node_);
      RRTNode3DPtr selected_SI = start_node_, selected_GI = goal_node_;
      // double min_houristic = h_start_goal;
      kdtree *treeA = kdtree_1;
      kdtree *treeB = kdtree_2;

      std::random_device rd;                                // Seed
      std::mt19937 gen(rd());                               // Mersenne Twister engine
      std::uniform_real_distribution<double> dis(0.0, 1.0); // Uniform distribution [0,1)

      /* main loop */
      number_of_iterations_ = 0;

#ifdef DEBUG
      std::cout << "[BRRT_Optimize] Start sampling..." << std::endl;
#endif
      cache.insert(start_node_, treeA, goal_node_, treeB, h_start_goal); // insert start and goal node to cache
      for (number_of_iterations_ = 0; number_of_iterations_ < max_iteration_; ++number_of_iterations_)
      {
        if (!map_ptr_->isStateValid(start_node_->x) && !map_ptr_->isStateValid(goal_node_->x))
        {
          break;
        }
        bool bias = true;
        Eigen::Vector3d x_rand;
        /* random sampling */
        Eigen::Vector3d x_new;
        double random01 = dis(gen);
        struct kdres *p_nearestA = nullptr, *p_nearestB = nullptr;
        RRTNode3DPtr nearest_nodeA, nearest_nodeB;
        double h_tmp;
        double pbias = 0;
        if (cache.popMinByTree(treeA, treeB, selected_SI, selected_GI, h_tmp))
        {
          // If cache is empty, select start and goal nodes
          pbias = computePbias(
              brrt_optimize_p1_,
              h_start_goal,
              selected_SI->x,
              selected_GI->x);
        }
        // 1. Sample
        if (random01 < pbias)
        {
          bias = true;
          x_rand = weightSample(selected_SI, selected_GI->x, treeA, false);
          // Eigen::Vector3d x_tmp = computeT(selected_SI->x, selected_GI->x, x_rand);
          nearest_nodeA = selected_SI;
          // x_new = steer(nearest_nodeA->x, x_tmp, steer_length_);
          // x_new = AFBGSteer(nearest_nodeA->x, x_rand, selected_GI->x, steer_length_);
          x_new = AFBGSteer(nearest_nodeA->x, x_rand, selected_GI->x, 0.0);

          if ((!map_ptr_->isStateValid(x_new)) || (!map_ptr_->isSegmentValid(nearest_nodeA->x, x_new)))
          {
            std::swap(treeA, treeB);
            path_reverse = !path_reverse;
            continue;
          }
          nearest_nodeB = selected_GI;
#ifdef DEBUG
          if (vis_ptr_)
          {
            // Pass the .x member which contains the Eigen::Vector3d coordinates
            vis_ptr_->visualize_a_ball(selected_SI->x, 0.5, "/brrt_optimize/SI", visualization::Color::red);
            vis_ptr_->visualize_a_ball(selected_GI->x, 0.5, "/brrt_optimize/GI", visualization::Color::blue);

            // Optional: Draw a line between the two guide nodes to see the "bridge"
            std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> guide_bridge;
            guide_bridge.push_back({selected_SI->x, selected_GI->x});
            vis_ptr_->visualize_pairline(guide_bridge, "/brrt_optimize/guide_pair_line", visualization::Color::yellow, 0.1);

            // std::cout << "[BRRT_Optimize] Use heuristic steer. Pbias: " << pbias << std::endl;
          }
#endif
        }
        else
        {
          bias = false;
          x_rand = get_sample_valid();
          // x_new = map_ptr_->getFreeNodeInLine(nearest_nodeA->x, x_rand, brrt_optimize_step_);
          // #ifdef DEBUG
          //           std::cout << "[BRRT_Optimize] Use normal steer " << pbias << std::endl;
          // #endif
          p_nearestA = kd_nearest3(treeA, x_rand[0], x_rand[1], x_rand[2]);

          if (p_nearestA == nullptr)
          {
#ifdef DEBUG
            ROS_ERROR("nearest query error");
#endif
            continue;
          }
          nearest_nodeA = (RRTNode3DPtr)kd_res_item_data(p_nearestA);
          kd_res_free(p_nearestA);
          // x_new = AFBGSteer(nearest_nodeA->x, x_rand, goal_node_->x, steer_length_);
          x_new = AFBGSteer(nearest_nodeA->x, x_rand, goal_node_->x, 0.0);
          // if ((!map_ptr_->isStateValid(x_new)) || (!map_ptr_->isSegmentValid(nearest_nodeA->x, x_new)))
          // {
          //   std::swap(treeA, treeB);
          //   path_reverse = !path_reverse;
          //   continue;
          // }
          // [MODIFIED] Ray Marching: Walk until blocked
          Eigen::Vector3d dir_vec = x_new - nearest_nodeA->x;
          double full_dist = dir_vec.norm();

          if (full_dist < 1e-4)
            continue; // Too small

          Eigen::Vector3d unit_dir = dir_vec.normalized();
          double resolution = map_ptr_->getResolution();
          if (resolution <= 0.001)
            resolution = 0.1; // Safety

          double curr_dist = 0.0;
          Eigen::Vector3d last_valid_pos = nearest_nodeA->x;
          bool hit_obstacle = false;

          while (curr_dist < full_dist)
          {
            curr_dist += resolution;
            if (curr_dist > full_dist)
              curr_dist = full_dist;

            Eigen::Vector3d test_pos = nearest_nodeA->x + unit_dir * curr_dist;

            if (!map_ptr_->isStateValid(test_pos))
            {
              hit_obstacle = true;
              break; // Stop at last valid pos
            }
            last_valid_pos = test_pos;
          }

          // Update x_new to whatever point we reached
          x_new = last_valid_pos;

          // If we barely moved (stuck), skip
          if ((x_new - nearest_nodeA->x).norm() < 0.1)
          {
            continue;
          }
          p_nearestB = kd_nearest3(treeB, x_new[0], x_new[1], x_new[2]);
          if (p_nearestB == nullptr)
          {
#ifdef DEBUG
            ROS_ERROR("nearest query error");
#endif
            continue;
          }
          nearest_nodeB = (RRTNode3DPtr)kd_res_item_data(p_nearestB);
          kd_res_free(p_nearestB);
        }

        // 4. Add Node
        double actual_dist = (x_new - nearest_nodeA->x).norm();
        double dist_from_A = nearest_nodeA->cost_from_start + actual_dist;
        RRTNode3DPtr new_nodeA(nullptr);
        if (valid_tree_node_nums_ + 1 >= max_tree_node_nums_)
        {
          valid_tree_node_nums_ = max_tree_node_nums_; // max_node_num reached
          break;
        }
        new_nodeA = addTreeNode(nearest_nodeA, x_new, dist_from_A, actual_dist);

#ifdef DEBUG
        if (vis_ptr_ && new_nodeA)
        {
          std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> new_edge;
          new_edge.push_back({nearest_nodeA->x, x_new});
          if (bias == true)
          {
            vis_ptr_->visualize_pairline(new_edge, "/brrt_optimize/step_edge_A", visualization::Color::green, 0.2);
            vis_ptr_->visualize_a_ball(x_new, 0.4, "/brrt_optimize/step_node_A", visualization::Color::green);
          }
          else
          {
            vis_ptr_->visualize_pairline(new_edge, "/brrt_optimize/step_edge_A", visualization::Color::yellow, 0.2);
            vis_ptr_->visualize_a_ball(x_new, 0.4, "/brrt_optimize/step_node_A", visualization::Color::yellow);
          }
          usleep(1000000); // 0.4s: Tree Growth
        }
#endif

        kd_insert3(treeA, x_new[0], x_new[1], x_new[2], new_nodeA);
        update_cache_nearest_heuristic(new_nodeA, treeA, treeB); // update cache with new node
        vector<Eigen::Vector3d> x_connects;
        // bool isConnected = greedySteer(nearest_nodeB->x, x_new, x_connects, steer_length_);

        double adaptive_connect_len = getAdaptiveSteerLength(nearest_nodeB->x);
        bool isConnected = greedySteer(nearest_nodeB->x, x_new, x_connects, adaptive_connect_len);
        
        /* Add the steered nodes to treeB */
        RRTNode3DPtr new_nodeB = nearest_nodeB;
        if (!x_connects.empty())
        {
          if (valid_tree_node_nums_ + (int)x_connects.size() >= max_tree_node_nums_)
          {
            valid_tree_node_nums_ = max_tree_node_nums_; // max_node_num reached
            break;
          }

          for (auto x_connect : x_connects)
          {
            new_nodeB = addTreeNode(new_nodeB, x_connect, new_nodeB->cost_from_start + steer_length_, steer_length_);

            kd_insert3(treeB, x_connect[0], x_connect[1], x_connect[2], new_nodeB);
            update_cache_nearest_heuristic(new_nodeB, treeB, treeA);
          }
        }

        /* If connected, trace the connected path */
        if (isConnected)
        {

          tree_connected = true;
          double path_cost = new_nodeA->cost_from_start + new_nodeB->cost_from_start + calDist(new_nodeB->x, new_nodeA->x);
          if (path_cost < cost_best_)
          {
            vector<Eigen::Vector3d> curr_best_path;
            if (path_reverse)
              fillPath(new_nodeB, new_nodeA, curr_best_path);
            else
              fillPath(new_nodeA, new_nodeB, curr_best_path);
            path_list_.emplace_back(curr_best_path);
            solution_cost_time_pair_list_.emplace_back(path_cost, (ros::Time::now() - rrt_start_time).toSec());
            cost_best_ = path_cost;
          }
#ifdef DEBUG
          std::cout << "[BRRT_Optimize]**********Find path after " << number_of_iterations_ << " iterations" << std::endl;
#endif
          break;
        }
        else
        {
          std::swap(treeA, treeB);
          path_reverse = !path_reverse;
        }

#ifdef DEBUG
        // std::cout << "[BRRT_Optimize] Iteration " << number_of_iterations_ << " completed, tree size: " << valid_tree_node_nums_ << std::endl;
        visualizeWholeTree();

        // vis_ptr_->visualize_a_ball(x_rand, 0.5, "/brrt_optimize/x_rand", visualization::Color::blue);
        // vis_ptr_->visualize_a_ball(x_new, 0.5, "/brrt_optimize/x_new", visualization::Color::green);
        // vis_ptr_->visualize_a_ball(nearest_nodeA->x, 0.5, "/brrt_optimize/nearest_nodeA", visualization::Color::black);
        // vis_ptr_->visualize_a_ball(nearest_nodeB->x, 0.5, "/brrt_optimize/nearest_nodeB", visualization::Color::white);
        // usleep(50000); // Sleep for 0.1 seconds to visualize the tree growth
#endif

      } // End of loop
#ifdef DEBUG
      visualizeWholeTree();
#endif
      final_path_use_time_ = (ros::Time::now() - rrt_start_time).toSec();
      if (tree_connected)
      {

#ifdef DEBUG
        ROS_INFO_STREAM("[BRRT_Optimize]: find_path_use_time: " << solution_cost_time_pair_list_.front().second << ", length: " << solution_cost_time_pair_list_.front().first);
#endif
        // vis_ptr_->visualize_a_text(Eigen::Vector3d(0, 0, 0), "find_path_use_time","find_path_use_time: " + std::to_string(solution_cost_time_pair_list_.front().second), visualization::Color::black);
        // vis_ptr_->visualize_a_text(Eigen::Vector3d(0, 0, 0.5), "length","length: " + std::to_string(solution_cost_time_pair_list_.front().first), visualization::Color::black);

        // visualizeWholeTree();
        final_path_ = path_list_.back();
      }
#ifdef DEBUG
      else if (valid_tree_node_nums_ == max_tree_node_nums_)
      {
        // visualizeWholeTree();
        ROS_ERROR_STREAM("[BRRT_Optimize]: NOT CONNECTED TO GOAL after " << max_tree_node_nums_ << " nodes added to rrt-tree");
      }
      else
      {
        ROS_ERROR_STREAM("[BRRT_Optimize]: NOT CONNECTED TO GOAL after " << (ros::Time::now() - rrt_start_time).toSec() << " seconds");
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
      vis_ptr_->visualize_balls(tree_nodes, "tree_vertice", visualization::Color::blue, 1.0);
      vis_ptr_->visualize_pairline(edges, "tree_edges", visualization::Color::red, 0.06);
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
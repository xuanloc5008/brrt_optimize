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
#include <random>
#include <cmath>
#include <limits>
#include <iostream> // [DEBUG] Added for std::cout

namespace path_plan
{
  class BRRT_Optimize_Case3
  {
  public:
    BRRT_Optimize_Case3() {};

    BRRT_Optimize_Case3(const ros::NodeHandle &nh, const env::OccMap::Ptr &mapPtr) : nh_(nh), map_ptr_(mapPtr)
    {
      nh_.param("BRRT/steer_length", steer_length_, 0.0);
      nh_.param("BRRT/search_time", search_time_, 0.0);
      nh_.param("BRRT/max_tree_node_nums", max_tree_node_nums_, 0);

      nh_.param("BRRT_Optimize/p1", brrt_optimize_p1_, 0.8);
      nh_.param("BRRT_Optimize/u_p", brrt_optimize_u_p, 2.0);
      nh_.param("BRRT_Optimize/step", brrt_optimize_step_, 0.1);

      nh_.param("BRRT_Optimize/alpha", brrt_optimize_alpha_, 0.5);
      nh_.param("BRRT_Optimize/beta", brrt_optimize_beta_, 0.3);
      nh_.param("BRRT_Optimize/gamma", brrt_optimize_gamma_, 0.5);
      nh_.param("BRRT_Optimize/max_iteration", max_iteration_, 0);
      nh_.param("BRRT_Optimize/enable2d", brrt_enable_2d, true);

      // ---- SOF-like parameters (sampling + steer) ----
      nh_.param("BRRT_Optimize/sof/epsilon_init", eps_init_, 1.0);
      nh_.param("BRRT_Optimize/sof/epsilon_floor", eps_floor_, 0.2);
      nh_.param("BRRT_Optimize/sof/epsilon_gamma", eps_decay_, 0.9991);
      nh_.param("BRRT_Optimize/sof/p_goal_sample", p_goal_sample_, 0.02);
      nh_.param("BRRT_Optimize/sof/p_global_uniform", p_global_uniform_, 0.05);

      nh_.param("BRRT_Optimize/sof/weight_blocks", weight_blocks_, 16);
      nh_.param("BRRT_Optimize/sof/weight_grade", weight_grade_, 1.0);
      nh_.param("BRRT_Optimize/sof/lidar_step", lidar_step_, 0.10);

      // ---- anytime optimization ----
      nh_.param("BRRT_Optimize/anytime", anytime_opt_, true);
      nh_.param("BRRT_Optimize/stagnation_limit", stagnation_limit_, 400);

      // ---- optional cache pair selection novelty ----
      nh_.param("BRRT_Optimize/use_boltzmann_pair", use_boltzmann_pair_, false);
      nh_.param("BRRT_Optimize/boltzmann_T_init", boltzmann_T_init_, 1.0);
      nh_.param("BRRT_Optimize/boltzmann_T_decay", boltzmann_T_decay_, 0.9995);
      nh_.param("BRRT_Optimize/boltzmann_T_min", boltzmann_T_min_, 0.05);

      // ---- cache update ----
      nh_.param("BRRT_Optimize/cache_k", cache_k_nearest_, 20);

      ROS_WARN_STREAM("[BRRT_Optimize_case3] param: steer_length: " << steer_length_);
      ROS_WARN_STREAM("[BRRT_Optimize_case3] param: search_time: " << search_time_);
      ROS_WARN_STREAM("[BRRT_Optimize_case3] param: max_tree_node_nums: " << max_tree_node_nums_);
      ROS_WARN_STREAM("[BRRT_Optimize_case3] param: anytime_opt: " << anytime_opt_);

      sampler_.setSamplingRange(mapPtr->getOrigin(), mapPtr->getMapSize());

      map_diag_ = mapPtr->getMapSize().norm();
      if (map_diag_ < 1e-3)
        map_diag_ = 50.0;

      valid_tree_node_nums_ = 0;
      nodes_pool_.resize(max_tree_node_nums_);
      for (int i = 0; i < max_tree_node_nums_; ++i)
      {
        nodes_pool_[i] = new TreeNode;
      }
    }

    ~BRRT_Optimize_Case3()
    {
      // safe cleanup in algorithm code (node.h unchanged)
      for (auto *p : nodes_pool_)
        delete p;
      nodes_pool_.clear();
    };

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

      valid_tree_node_nums_ = 2; // put start and goal in pool
      cache.clear();             // clear the heuristic cache before planning

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

    void set_heuristic_param(double p1, double u_p, double alpha, double beta, double gamma, double steer_length)
    {
      brrt_optimize_p1_ = p1;
      brrt_optimize_u_p = u_p;
      brrt_optimize_alpha_ = alpha;
      brrt_optimize_beta_ = beta;
      brrt_optimize_gamma_ = gamma;
      steer_length_ = steer_length;
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

#ifdef DEBUG
    void print_vector3d(std::string name, Eigen::Vector3d &p)
    {
      std::cout << name << " x: " << p[0] << " y: " << p[1] << " z: " << p[2] << std::endl;
    }
#endif

  private:
    // nodehandle params
    ros::NodeHandle nh_;

    double rewire_radius_init_ = 5.0;
    BiasSampler sampler_;

    // original params
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

    // SOF-like params
    double eps_init_{1.0};
    double eps_floor_{0.2};
    double eps_decay_{0.9991};
    double p_goal_sample_{0.02};
    double p_global_uniform_{0.05};
    int weight_blocks_{16};
    double weight_grade_{1.0};
    double lidar_step_{0.1};

    bool anytime_opt_{true};
    int stagnation_limit_{400};

    bool use_boltzmann_pair_{false};
    double boltzmann_T_init_{1.0};
    double boltzmann_T_decay_{0.9995};
    double boltzmann_T_min_{0.05};

    int cache_k_nearest_{20};

    double map_diag_{50.0};

    // internal state
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

    // RNG (persistent)
    std::mt19937 rng_{std::random_device{}()};
    std::uniform_real_distribution<double> uni01_{0.0, 1.0};

  private:
    static inline double clamp01(double v)
    {
      if (v < 0.0)
        return 0.0;
      if (v > 1.0)
        return 1.0;
      return v;
    }

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

    RRTNode3DPtr addTreeNode(RRTNode3DPtr &parent,
                             const Eigen::Vector3d &state,
                             const double &cost_from_start,
                             const double &cost_from_parent)
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

      // update descendants cost_from_start
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
      if (dist <= len)
        return rand_node_p;
      return nearest_node_p + diff_vec * len / dist;
    }

    bool greedySteer(const Eigen::Vector3d &x_near,
                     const Eigen::Vector3d &x_target,
                     vector<Eigen::Vector3d> &x_connects,
                     const double len)
    {
      double vec_length = (x_target - x_near).norm();
      x_connects.clear();

      // avoid div by zero
      if (vec_length < 1e-8)
        return true;

      if (vec_length < len)
        return map_ptr_->isSegmentValid(x_near, x_target);

      Eigen::Vector3d vec_unit = (x_target - x_near) / vec_length;

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
      Eigen::Vector3d si_gi = si - gi;
      Eigen::Vector3d si_G = si - goal_node_->x;
      Eigen::Vector3d gi_S = gi - start_node_->x;

      double si_gi_dist = si_gi.norm();
      double si_G_dist = si_G.norm();
      double gi_S_dist = gi_S.norm();

      return brrt_optimize_alpha_ * si_gi_dist + brrt_optimize_beta_ * si_G_dist + brrt_optimize_gamma_ * gi_S_dist;
    }

    // cache update: insert heuristic from nodeSi to K nearest in other tree
    void update_cache_nearest_heuristic(RRTNode3DPtr nodeSi, kdtree *treeA, kdtree *treeB)
    {
      struct kdres *nodesB = kd_nearest_n(treeB, nodeSi->x.data(), std::max(1, cache_k_nearest_));
      while (nodesB && !kd_res_end(nodesB))
      {
        RRTNode3DPtr nodeGi = (RRTNode3DPtr)kd_res_item_data(nodesB);
        double h = computeH(nodeSi->x, nodeGi->x);
        cache.insert(nodeSi, treeA, nodeGi, treeB, h);
        kd_res_next(nodesB);
      }
      if (nodesB)
        kd_res_free(nodesB);
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

    static inline double sigmoid01(double x)
    {
      return 1.0 / (1.0 + std::exp(-x));
    }

    // Approximate nearest obstacle distance by ray marching in XY (SOF-style dist(.) proxy)
    double approxObstacleDist2D(const Eigen::Vector3d &x, double max_range)
    {
      double best = max_range;
      int N = std::max(8, weight_blocks_);
      for (int i = 0; i < N; ++i)
      {
        double ang = (2.0 * M_PI * i) / (double)N;
        Eigen::Vector3d dir(std::cos(ang), std::sin(ang), 0.0);

        for (double r = lidar_step_; r <= max_range; r += lidar_step_)
        {
          Eigen::Vector3d p = x + r * dir;
          p.z() = x.z();
          if (!map_ptr_->isStateValid(p))
          {
            if (r < best)
              best = r;
            break;
          }
        }
      }
      return best;
    }

    // AFBG-like steer: goal bias + obstacle tangential bias (2D-friendly, works in 3D by keeping z)
    Eigen::Vector3d afbgSteer(const Eigen::Vector3d &x_near,
                              const Eigen::Vector3d &x_rand,
                              const Eigen::Vector3d &x_goal_like,
                              double delta)
    {
      Eigen::Vector3d near = x_near;
      Eigen::Vector3d rand = x_rand;
      Eigen::Vector3d goal = x_goal_like;

      if (brrt_enable_2d)
      {
        rand.z() = near.z();
        goal.z() = near.z();
      }

      Eigen::Vector3d d_rand = rand - near;
      double nr = d_rand.norm();
      if (nr < 1e-9)
        return near;
      d_rand /= nr;

      Eigen::Vector3d d_goal = goal - near;
      double ng = d_goal.norm();
      if (ng < 1e-9)
        d_goal = d_rand;
      else
        d_goal /= ng;

      // goal bias factor φ (sigmoid-scaled)
      double phi = delta * sigmoid01((ng / std::max(1e-6, map_diag_)) * 5.0);

      // obstacle tangential bias η (stronger when closer to obstacles)
      double obsDist = approxObstacleDist2D(near, 2.0 * delta);
      double ratio = obsDist / std::max(1e-6, 2.0 * delta);
      double eta = delta * (1.0 - sigmoid01(ratio * 5.0));

      // tangent direction candidates
      Eigen::Vector3d tangent;
      if (brrt_enable_2d)
      {
        tangent = Eigen::Vector3d(-d_rand.y(), d_rand.x(), 0.0);
      }
      else
      {
        // any perpendicular direction (fallback)
        Eigen::Vector3d axis = (std::fabs(d_rand.z()) < 0.9) ? Eigen::Vector3d(0, 0, 1) : Eigen::Vector3d(1, 0, 0);
        tangent = d_rand.cross(axis);
        if (tangent.norm() < 1e-9)
          tangent = Eigen::Vector3d(-d_rand.y(), d_rand.x(), 0.0);
        else
          tangent.normalize();
      }

      // try both tangent signs and choose a valid one
      Eigen::Vector3d dir1 = d_rand + (phi / delta) * d_goal + (eta / delta) * tangent;
      Eigen::Vector3d dir2 = d_rand + (phi / delta) * d_goal - (eta / delta) * tangent;
      if (dir1.norm() < 1e-9)
        dir1 = d_rand;
      if (dir2.norm() < 1e-9)
        dir2 = d_rand;
      dir1.normalize();
      dir2.normalize();

      Eigen::Vector3d x1 = near + delta * dir1;
      Eigen::Vector3d x2 = near + delta * dir2;
      if (brrt_enable_2d)
      {
        x1.z() = near.z();
        x2.z() = near.z();
      }

      bool v1 = map_ptr_->isStateValid(x1) && map_ptr_->isSegmentValid(near, x1);
      bool v2 = map_ptr_->isStateValid(x2) && map_ptr_->isSegmentValid(near, x2);

      if (v1)
        return x1;
      if (v2)
        return x2;

      // fallback
      Eigen::Vector3d x3 = steer(near, rand, delta);
      if (brrt_enable_2d)
        x3.z() = near.z();
      return x3;
    }

    double computePbias(double Pinit,
                        double h_start_goal,
                        const Eigen::Vector3d &sguide,
                        const Eigen::Vector3d &tguide)
    {
      if (h_start_goal == 0.0 || brrt_optimize_u_p <= 0.00001)
      {
        return Pinit;
      }
      double h_sguide_tguide = computeH(sguide, tguide);
      double ratio = brrt_optimize_u_p * (h_start_goal - h_sguide_tguide) / h_start_goal;
      double Pbias = Pinit * std::exp(-ratio);
      return clamp01(Pbias);
    }

    // Your sector sampler (kept; RNG changed to member rng_)
    Eigen::Vector3d smartSectorSampling(const Eigen::Vector3d &A, const Eigen::Vector3d &B)
    {
      Eigen::Vector3d midpoint = (A + B) / 2.0;
      Eigen::Vector3d diff = B - A;
      double dist_AB = diff.norm();
      double radius = dist_AB / 2.0;

      if (radius < 0.05)
        return midpoint;

      Eigen::Vector3d normal = diff.normalized();
      Eigen::Vector3d u;

      if (std::abs(normal.x()) < 1e-6 && std::abs(normal.y()) < 1e-6)
        u = Eigen::Vector3d(0, 1, 0).cross(normal).normalized();
      else
        u = Eigen::Vector3d(0, 0, 1).cross(normal).normalized();

      Eigen::Vector3d v = normal.cross(u);

      const int num_sectors = 8;
      const int rays_per_sector = 5;
      const int steps_per_ray = 5;

      std::vector<std::pair<double, int>> candidates;
      candidates.reserve(num_sectors);

      std::vector<int> priority_sectors = {1, 2, 5, 6};
      std::vector<int> secondary_sectors = {0, 3, 4, 7};

      auto scan_sector_group = [&](const std::vector<int> &indices)
      {
        for (int i : indices)
        {
          double theta_start = i * (2 * M_PI / num_sectors);
          double theta_step = (2 * M_PI / num_sectors) / rays_per_sector;

          int obstacle_hits = 0;
          int total_checks = 0;

          for (int r = 0; r < rays_per_sector; ++r)
          {
            double ray_angle = theta_start + theta_step * (r + 0.5);
            Eigen::Vector3d ray_dir = std::cos(ray_angle) * u + std::sin(ray_angle) * v;

            for (int s = 1; s <= steps_per_ray; ++s)
            {
              double d = radius * ((double)s / steps_per_ray);
              Eigen::Vector3d check_point = midpoint + d * ray_dir;

              total_checks++;
              if (!map_ptr_->isStateValid(check_point))
                obstacle_hits++;
            }
          }
          double obs_ratio = (total_checks > 0) ? (double)obstacle_hits / total_checks : 0.0;
          candidates.push_back({obs_ratio, i});
        }
      };

      scan_sector_group(priority_sectors);
      std::sort(candidates.begin(), candidates.end());

      if (!candidates.empty() && candidates[0].first > 0.7)
      {
        scan_sector_group(secondary_sectors);
        std::sort(candidates.begin(), candidates.end());
      }

      if (candidates.empty())
        return midpoint;

      int chosen_sector_idx = candidates[0].second;
      if (candidates[0].first > 0.95)
        return midpoint;

      std::uniform_real_distribution<> dist01(0.0, 1.0);

      double angle_step = 2 * M_PI / num_sectors;
      double theta_base = chosen_sector_idx * angle_step;

      double sample_theta = theta_base + dist01(rng_) * angle_step;
      double sample_r = radius * std::sqrt(dist01(rng_));

      Eigen::Vector3d p = midpoint + sample_r * (std::cos(sample_theta) * u + std::sin(sample_theta) * v);
      return p;
    }

    // SOF-like spatial probability-weight sampling (uses member rng_, keeps z)
    Eigen::Vector3d spatialProbabilityWeightSampling(const Eigen::Vector3d &center_point, double max_range)
    {
      const int N_block = std::max(8, weight_blocks_);
      const double weightGrade = std::max(0.0, weight_grade_);

      std::vector<double> weights;
      std::vector<double> ray_lengths;
      weights.reserve(N_block);
      ray_lengths.reserve(N_block);

      double sum_weight = 0.0;
      for (int i = 0; i < N_block; ++i)
      {
        double angle = i * (2 * M_PI / N_block);
        Eigen::Vector3d direction(std::cos(angle), std::sin(angle), 0.0);

        double r = 0.0;
        for (; r <= max_range; r += lidar_step_)
        {
          Eigen::Vector3d check_pt = center_point + direction * r;
          check_pt.z() = center_point.z();
          if (!map_ptr_->isStateValid(check_pt))
          {
            break;
          }
        }

        ray_lengths.push_back(r);

        double w = std::pow(std::max(0.0, r), weightGrade);
        weights.push_back(w);
        sum_weight += w;
      }

      if (sum_weight < 1e-6)
        return get_sample_valid();

      std::uniform_real_distribution<> distW(0.0, sum_weight);
      double random_val = distW(rng_);

      int chosen_block_idx = 0;
      double current_sum = 0.0;
      for (int i = 0; i < N_block; ++i)
      {
        current_sum += weights[i];
        if (random_val <= current_sum)
        {
          chosen_block_idx = i;
          break;
        }
      }

      std::uniform_real_distribution<> dist01(0.0, 1.0);

      double angle_start = chosen_block_idx * (2 * M_PI / N_block);
      double angle_step = 2 * M_PI / N_block;

      double theta = angle_start + dist01(rng_) * angle_step;

      double r_limit = ray_lengths[chosen_block_idx];
      double r_sample = r_limit * std::sqrt(dist01(rng_));

      Eigen::Vector3d p = center_point + Eigen::Vector3d(r_sample * std::cos(theta), r_sample * std::sin(theta), 0.0);
      p.z() = center_point.z();
      return p;
    }

    // SOF Eq.(12)-style shrinking rewire radius (algorithm-only)
    double getAdaptiveRewireRadius(int tree_size)
    {
      if (tree_size <= 2)
        return rewire_radius_init_;

      double n = (double)tree_size;
      double shrink = std::sqrt(std::log10(std::max(2.0, n)) / std::max(2.0, n));
      double r = rewire_radius_init_ * shrink;

      return std::max(r, steer_length_ * 1.5);
    }

    void rewire(RRTNode3DPtr &new_node, kdtree *tree_ptr, int tree_size)
    {
      double r_near = getAdaptiveRewireRadius(tree_size);
      struct kdres *neighbors = kd_nearest_range3(tree_ptr, new_node->x[0], new_node->x[1], new_node->x[2], r_near);

      if (!neighbors || kd_res_size(neighbors) <= 1)
      {
        if (neighbors)
          kd_res_free(neighbors);
        return;
      }

      std::vector<RRTNode3DPtr> neighbor_nodes;
      while (!kd_res_end(neighbors))
      {
        RRTNode3DPtr nb = (RRTNode3DPtr)kd_res_item_data(neighbors);
        if (nb != new_node && nb != new_node->parent)
        {
          neighbor_nodes.push_back(nb);
        }
        kd_res_next(neighbors);
      }
      kd_res_free(neighbors);

      // try better parent for new_node
      for (auto &nb : neighbor_nodes)
      {
        double dist = calDist(nb->x, new_node->x);
        double new_cost = nb->cost_from_start + dist;

        if (new_cost + 1e-9 < new_node->cost_from_start)
        {
          if (map_ptr_->isSegmentValid(nb->x, new_node->x))
          {
            changeNodeParent(new_node, nb, dist);
          }
        }
      }

      // try rewire neighbors to new_node
      for (auto &nb : neighbor_nodes)
      {
        double dist = calDist(new_node->x, nb->x);
        double new_cost = new_node->cost_from_start + dist;

        if (new_cost + 1e-9 < nb->cost_from_start)
        {
          if (map_ptr_->isSegmentValid(new_node->x, nb->x))
          {
            changeNodeParent(nb, new_node, dist);
          }
        }
      }
    }

    bool brrt_optimize(const Eigen::Vector3d &s, const Eigen::Vector3d &g)
    {
      ros::Time rrt_start_time = ros::Time::now();

      bool tree_connected = false;
      bool path_reverse = false;

      // init heuristic
      double h_start_goal = computeH(start_node_->x, goal_node_->x);

      // kd tree init
      kdtree *kdtree_1 = kd_create(3);
      kdtree *kdtree_2 = kd_create(3);
      if (!kdtree_1 || !kdtree_2)
        return false;

      kd_insert3(kdtree_1, start_node_->x[0], start_node_->x[1], start_node_->x[2], start_node_);
      kd_insert3(kdtree_2, goal_node_->x[0], goal_node_->x[1], goal_node_->x[2], goal_node_);

      kdtree *treeA = kdtree_1;
      kdtree *treeB = kdtree_2;

      // keep vectors for epsilon-greedy random node selection
      std::vector<RRTNode3DPtr> vecA, vecB;
      vecA.reserve(max_tree_node_nums_ / 2);
      vecB.reserve(max_tree_node_nums_ / 2);
      vecA.push_back(start_node_);
      vecB.push_back(goal_node_);
      int sizeA = 1, sizeB = 1;

      // initial cache insert
      cache.insert(start_node_, treeA, goal_node_, treeB, h_start_goal);

      // SOF schedules
      double eps = eps_init_;
      double temperature = boltzmann_T_init_;
      int stagnation = 0;

      number_of_iterations_ = 0;
#ifdef DEBUG
      std::cout << "[BRRT_Optimize_case3] Start sampling..." << std::endl;
#endif

      for (number_of_iterations_ = 0; number_of_iterations_ < max_iteration_; ++number_of_iterations_)
      {
        // time budget
        if (search_time_ > 1e-6 && (ros::Time::now() - rrt_start_time).toSec() > search_time_)
          break;

        // update schedules (SOF epsilon decay)
        eps = std::max(eps * eps_decay_, eps_floor_);
        temperature = std::max(temperature * boltzmann_T_decay_, boltzmann_T_min_);

        // choose guide pair from cache
        RRTNode3DPtr selected_SI = nullptr;
        RRTNode3DPtr selected_GI = nullptr;
        double h_tmp = std::numeric_limits<double>::infinity();

        bool has_pair = false;
        if (use_boltzmann_pair_)
          has_pair = cache.getBoltzmannPair(treeA, treeB, selected_SI, selected_GI, h_tmp, temperature);
        else
          has_pair = cache.getMinByTree(treeA, treeB, selected_SI, selected_GI, h_tmp);

        if (!has_pair || selected_SI == nullptr || selected_GI == nullptr)
        {
          // fallback
          selected_SI = vecA.front();
          selected_GI = vecB.front();
        }

        // compute pair-guided probability (keep your mechanism)
        double pbias_pair = computePbias(brrt_optimize_p1_, h_start_goal, selected_SI->x, selected_GI->x);

        // sampling mode selection
        double r01 = uni01_(rng_);

        RRTNode3DPtr nearest_nodeA = nullptr, nearest_nodeB = nullptr;
        Eigen::Vector3d x_rand, x_new;

        // "target" for biasing this expansion: connect toward opposite guide node
        Eigen::Vector3d x_target = selected_GI->x;

        // -------- Mode 1: Cache-pair guided sector sampling --------
        if (r01 < pbias_pair)
        {
          x_rand = smartSectorSampling(selected_SI->x, selected_GI->x);
#ifdef DEBUG
          if (vis_ptr_)
            vis_ptr_->visualize_a_ball(x_rand, 0.5, "/brrt_optimize/x_tmp", visualization::Color::red);
#endif
          nearest_nodeA = selected_SI;
          nearest_nodeB = selected_GI;

          x_new = afbgSteer(nearest_nodeA->x, x_rand, x_target, steer_length_);

          if ((!map_ptr_->isStateValid(x_new)) || (!map_ptr_->isSegmentValid(nearest_nodeA->x, x_new)))
          {
            std::swap(treeA, treeB);
            std::swap(vecA, vecB);
            std::swap(sizeA, sizeB);
            path_reverse = !path_reverse;
            continue;
          }
        }
        // -------- Mode 2: SOF-like epsilon-greedy + weight sampling / global fallback --------
        else
        {
          // small probability: global uniform (keeps exploration robust)
          if (uni01_(rng_) < p_global_uniform_)
          {
            x_rand = get_sample_valid();
          }
          else
          {
            // goal sample (SOF p_goal)
            if (uni01_(rng_) < p_goal_sample_)
            {
              x_rand = x_target;
#ifdef DEBUG
              if (vis_ptr_)
                vis_ptr_->visualize_a_ball(x_rand, 0.5, "/brrt_optimize/x_tmp", visualization::Color::red);
#endif
            }
            else
            {
              // epsilon-greedy chooseNode: random node or nearest-to-target node
              RRTNode3DPtr chooseNode = nullptr;
              if (uni01_(rng_) < eps && !vecA.empty())
              {
                std::uniform_int_distribution<int> ridx(0, (int)vecA.size() - 1);
                chooseNode = vecA[ridx(rng_)];
              }
              else
              {
                // exploit: nearest in treeA to target
                struct kdres *p = kd_nearest3(treeA, x_target[0], x_target[1], x_target[2]);
                if (p)
                {
                  chooseNode = (RRTNode3DPtr)kd_res_item_data(p);
                  kd_res_free(p);
                }
                if (!chooseNode)
                  chooseNode = selected_SI;
              }

              x_rand = spatialProbabilityWeightSampling(chooseNode->x, steer_length_ * 3.0);
            }
          }

          // nearest in treeA to x_rand
          struct kdres *p_nearestA = kd_nearest3(treeA, x_rand[0], x_rand[1], x_rand[2]);
          if (!p_nearestA)
          {
#ifdef DEBUG
            ROS_ERROR("nearest query error");
#endif
            std::swap(treeA, treeB);
            std::swap(vecA, vecB);
            std::swap(sizeA, sizeB);
            path_reverse = !path_reverse;
            continue;
          }
          nearest_nodeA = (RRTNode3DPtr)kd_res_item_data(p_nearestA);
          kd_res_free(p_nearestA);

          x_new = afbgSteer(nearest_nodeA->x, x_rand, x_target, steer_length_);

          if ((!map_ptr_->isStateValid(x_new)) || (!map_ptr_->isSegmentValid(nearest_nodeA->x, x_new)))
          {
            std::swap(treeA, treeB);
            std::swap(vecA, vecB);
            std::swap(sizeA, sizeB);
            path_reverse = !path_reverse;
            continue;
          }

          // nearest in treeB to x_new
          struct kdres *p_nearestB = kd_nearest3(treeB, x_new[0], x_new[1], x_new[2]);
          if (!p_nearestB)
          {
#ifdef DEBUG
            ROS_ERROR("nearest query error");
#endif
            std::swap(treeA, treeB);
            std::swap(vecA, vecB);
            std::swap(sizeA, sizeB);
            path_reverse = !path_reverse;
            continue;
          }
          nearest_nodeB = (RRTNode3DPtr)kd_res_item_data(p_nearestB);
          kd_res_free(p_nearestB);
        }

        // Add new node to treeA (use true edge length)
        if (valid_tree_node_nums_ + 1 >= max_tree_node_nums_)
        {
          valid_tree_node_nums_ = max_tree_node_nums_;
          break;
        }

        double edgeA = calDist(nearest_nodeA->x, x_new);
        RRTNode3DPtr new_nodeA = addTreeNode(nearest_nodeA, x_new,
                                             nearest_nodeA->cost_from_start + edgeA, edgeA);

        kd_insert3(treeA, x_new[0], x_new[1], x_new[2], new_nodeA);
        vecA.push_back(new_nodeA);
        sizeA++;

        update_cache_nearest_heuristic(new_nodeA, treeA, treeB);
        rewire(new_nodeA, treeA, sizeA);

        // Greedy connect from treeB toward x_new
        vector<Eigen::Vector3d> x_connects;
        bool isConnected = greedySteer(nearest_nodeB->x, x_new, x_connects, steer_length_);

        RRTNode3DPtr new_nodeB = nearest_nodeB;
        if (!x_connects.empty())
        {
          if (valid_tree_node_nums_ + (int)x_connects.size() >= max_tree_node_nums_)
          {
            valid_tree_node_nums_ = max_tree_node_nums_;
            break;
          }

          Eigen::Vector3d prev = nearest_nodeB->x;
          for (auto &x_connect : x_connects)
          {
            double edgeB = calDist(prev, x_connect);
            new_nodeB = addTreeNode(new_nodeB, x_connect,
                                    new_nodeB->cost_from_start + edgeB, edgeB);
            kd_insert3(treeB, x_connect[0], x_connect[1], x_connect[2], new_nodeB);

            vecB.push_back(new_nodeB);
            sizeB++;

            rewire(new_nodeB, treeB, sizeB);
            prev = x_connect;
          }
          update_cache_nearest_heuristic(new_nodeB, treeB, treeA);
        }

        if (isConnected)
        {
          tree_connected = true;

          double connect_edge = calDist(new_nodeB->x, new_nodeA->x);
          double path_cost = new_nodeA->cost_from_start + new_nodeB->cost_from_start + connect_edge;

          if (path_cost + 1e-9 < cost_best_)
          {
            vector<Eigen::Vector3d> curr_best_path;
            if (path_reverse)
              fillPath(new_nodeB, new_nodeA, curr_best_path);
            else
              fillPath(new_nodeA, new_nodeB, curr_best_path);

            path_list_.emplace_back(curr_best_path);
            solution_cost_time_pair_list_.emplace_back(path_cost, (ros::Time::now() - rrt_start_time).toSec());
            cost_best_ = path_cost;
            stagnation = 0;
#ifdef DEBUG
            std::cout << "[BRRT_Optimize_case3]**********Find path after " << number_of_iterations_ << " iterations" << std::endl;
#endif
          }
          else
          {
            stagnation++;
          }

          // anytime or stop at first
          if (!anytime_opt_)
            break;
          if (stagnation > stagnation_limit_)
            break;
        }

#ifdef DEBUG
        // Optional: Visualize every iteration (Slows down performance significantly!)
        // visualizeWholeTree();
#endif

        // Alternate expansion: swap trees
        std::swap(treeA, treeB);
        std::swap(vecA, vecB);
        std::swap(sizeA, sizeB);
        path_reverse = !path_reverse;
      }

#ifdef DEBUG
      visualizeWholeTree();
#endif

      final_path_use_time_ = (ros::Time::now() - rrt_start_time).toSec();

      if (tree_connected && !path_list_.empty())
      {
        final_path_ = path_list_.back();
#ifdef DEBUG
        ROS_INFO_STREAM("[BRRT_Optimize_case3]: find_path_use_time: " << solution_cost_time_pair_list_.front().second << ", length: " << solution_cost_time_pair_list_.front().first);
#endif
      }
#ifdef DEBUG
      else if (valid_tree_node_nums_ >= max_tree_node_nums_)
      {
        ROS_ERROR_STREAM("[BRRT_Optimize_case3]: NOT CONNECTED TO GOAL after " << max_tree_node_nums_ << " nodes added to rrt-tree");
      }
      else
      {
        ROS_ERROR_STREAM("[BRRT_Optimize_case3]: NOT CONNECTED TO GOAL after " << (ros::Time::now() - rrt_start_time).toSec() << " seconds");
      }
#endif

      // free kdtrees (algorithm code only)
      kd_free(kdtree_1);
      kd_free(kdtree_2);

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

      // bfs
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
    // preserved sampling (for benchmarking)
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
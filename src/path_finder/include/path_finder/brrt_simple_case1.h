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
CONTRACT, STRICT LIABILITY, OR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE.
*/
#ifndef BRRT_SIMPLE_CASE1_H
#define BRRT_SIMPLE_CASE1_H
#include <string>
#include "occ_grid/occ_map.h"
#include "visualization/visualization.hpp"
#include "sampler.h"
#include "node.h"
#include "kdtree.h"
// #include "custom_logger.h"
#include <ros/ros.h>
#include <utility>
#include <queue>
#include <algorithm>
#include "nlohmann/json.hpp" // Assumes nlohmann/json.hpp is in your include path

// NEW: Include the message header for the percentage
#include <std_msgs/Float64.h>

double smallestDis = 5.0;
int count_trap = 0;

using json = nlohmann::json;
using std::string;
namespace path_plan
{
  class BRRT_Simple_Case1
  {
  public:
    BRRT_Simple_Case1() {};
    BRRT_Simple_Case1(const ros::NodeHandle &nh, const env::OccMap::Ptr &mapPtr) : nh_(nh), map_ptr_(mapPtr)
    {
      nh_.param("BRRT/steer_length", steer_length_, 0.0);
      nh_.param("BRRT/search_time", search_time_, 0.0);
      nh_.param("BRRT/max_tree_node_nums", max_tree_node_nums_, 0);

      nh_.param("BRRT_Optimize/p1", brrt_optimize_p1_, 0.8);
      nh_.param("BRRT_Optimize/u_p", brrt_optimize_u_p, 0.5);
      nh_.param("BRRT_Optimize/step", brrt_optimize_step_, 0.1);

      nh_.param("BRRT_Optimize/alpha", brrt_optimize_alpha_, 0.5);
      nh_.param("BRRT_Optimize/sampling_log_file", sampling_log_file_, std::string("/home/phuong/DACN/ICIT/brrt_optimize/src/path_finder/include/path_finder/sampling_log.jsonl"));
      sampling_log_stream_.open(sampling_log_file_, std::ios::out | std::ios::app);
      if (!sampling_log_stream_.is_open())
      {
        ROS_ERROR_STREAM("[BRRT_Simple_Case1] Could not open sampling log file: " << sampling_log_file_);
      }
      nh_.param("BRRT_Optimize/beta", brrt_optimize_beta_, 0.3);
      nh_.param("BRRT_Optimize/gamma", brrt_optimize_gamma_, 0.5);
      nh_.param("BRRT_Optimize/max_iteration", max_iteration_, 0);
      nh_.param("BRRT/enable2d", brrt_enable_2d, true);

      // Optional slow-mo visualization; default 0 to avoid stalling long runs.
      nh_.param("BRRT_Optimize/step_by_step_delay", step_delay_, 0.0);
      if (step_delay_ > 0.0)
      {
        ROS_WARN("[BRRT_Simple_Case1] Step-by-step visualization enabled with delay: %f s", step_delay_);
      }

      // --- Parameters for dynamic trap limit ---
      nh_.param("BRRT_Optimize/base_trap_limit", base_trap_limit_, 10);
      nh_.param("BRRT_Optimize/trap_limit_scaling_factor", trap_limit_scaling_factor_, 0.5);
      TRAP_COUNT_LIMIT_ = base_trap_limit_; // Initialize with the base value

      std::string percentage_topic;
      nh_.param<std::string>("BRRT_Optimize/obstacle_percentage_topic", percentage_topic, "/map_obstacle_percentage");

      obstacle_percentage_sub_ = nh_.subscribe(percentage_topic, 1, &BRRT_Simple_Case1::obstaclePercentageCallback, this);
      ROS_INFO("[BRRT_Simple_Case1] Subscribing to obstacle percentage on topic: %s", obstacle_percentage_sub_.getTopic().c_str());
      ROS_INFO("[BRRT_Simple_Case1] Initial TRAP_COUNT_LIMIT set to base: %d", TRAP_COUNT_LIMIT_);

      // --- NEW: Parameters for dynamic sampling probability ---
      nh_.param("BRRT_Optimize/trap_limit_penalty_factor", trap_limit_penalty_factor_, 0.5);
      nh_.param("BRRT_Optimize/progress_boost_factor", progress_boost_factor_, 0.2);
      nh_.param("BRRT_Optimize/progress_threshold", H_PROGRESS_THRESHOLD_, 1.0);
      // --- END NEW ---

      ROS_WARN_STREAM("[BRRT_Optimize_case1] param: steer_length: " << steer_length_);
      ROS_WARN_STREAM("[BRRT_Optimize_case1] param: search_time: " << search_time_);
      ROS_WARN_STREAM("[BRRT_Optimize_case1] param: max_tree_node_nums: " << max_tree_node_nums_);

      sampler_.setSamplingRange(mapPtr->getOrigin(), mapPtr->getMapSize());

      valid_tree_node_nums_ = 0;
      nodes_pool_.resize(max_tree_node_nums_);
      for (int i = 0; i < max_tree_node_nums_; ++i)
      {
        nodes_pool_[i] = new TreeNode;
      }
    }
    ~BRRT_Simple_Case1() {};

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

    Eigen::Vector3d get_sample_valid()
    {
      if (in_trap_mode_)
      {
        Eigen::Vector3d s = sampleOutsideTrap();
        if (map_ptr_->isStateValid(s))
          return s;
      }

      Eigen::Vector3d x_rand;
      int tries = 0;
      const int MAX_TRIES = 2000;
      sampler_.samplingOnce(x_rand);
      while (!map_ptr_->isStateValid(x_rand) ||
             (in_trap_mode_ && ((x_rand - trap_center_).norm() <= trap_radius_)))
      {
        sampler_.samplingOnce(x_rand);
        if (++tries > MAX_TRIES)
        {
          break;
        }
      }
      return x_rand;
    }

    Eigen::Vector3d sampleOutsideTrap()
    {
      if (!map_ptr_)
        return Eigen::Vector3d::Zero();

      // bounding box from map
      Eigen::Vector3d origin = map_ptr_->getOrigin();
      Eigen::Vector3d map_size = map_ptr_->getMapSize();
      double minx = origin[0], miny = origin[1], minz = origin[2];
      double maxx = minx + map_size[0], maxy = miny + map_size[1], maxz = minz + map_size[2];

      // compute maximum radius to reach corners of the bounding box
      std::array<Eigen::Vector3d, 8> corners = {
          Eigen::Vector3d(minx, miny, minz),
          Eigen::Vector3d(minx, miny, maxz),
          Eigen::Vector3d(minx, maxy, minz),
          Eigen::Vector3d(minx, maxy, maxz),
          Eigen::Vector3d(maxx, miny, minz),
          Eigen::Vector3d(maxx, miny, maxz),
          Eigen::Vector3d(maxx, maxy, minz),
          Eigen::Vector3d(maxx, maxy, maxz),
      };
      double maxR = 0.0;
      for (auto &c : corners)
        maxR = std::max(maxR, (c - trap_center_).norm());

      double r_min = std::max(trap_radius_ + 0.3, 0.5); // leave small margin
      double r_max = std::max(r_min + 1.0, maxR);
      std::random_device rd;
      static thread_local std::mt19937 gen(rd());
      std::uniform_real_distribution<double> u01(0.0, 1.0);
      std::uniform_real_distribution<double> uz(minz, maxz);

      const int MAX_ATTEMPTS = 200;
      for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt)
      {
        double u = u01(gen);
        double r = std::sqrt((1.0 - u) * (r_max * r_max) + u * (r_min * r_min));
        double theta = u01(gen) * 2.0 * M_PI;
        double x = trap_center_[0] + r * std::cos(theta);
        double y = trap_center_[1] + r * std::sin(theta);
        double z = uz(gen);

        x = std::min(std::max(x, minx), maxx);
        y = std::min(std::max(y, miny), maxy);
        z = std::min(std::max(z, minz), maxz);

        Eigen::Vector3d s(x, y, z);
        if ((s - trap_center_).norm() > trap_radius_ + 1e-6 && map_ptr_->isStateValid(s))
          return s;
      }

      Eigen::Vector3d fallback;
      sampler_.samplingOnce(fallback);
      return fallback;
    }

  private:
    // nodehandle params
    ros::NodeHandle nh_;

    ros::Subscriber obstacle_percentage_sub_;

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
    double step_delay_ = 0.0;

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

    std::string sampling_log_file_;
    std::ofstream sampling_log_stream_;

    // Trap detection members
    double h_past_ = DBL_MAX;
    int trapCount_ = 0;
    const double H_THRESHOLD_ = 0.2;

    int TRAP_COUNT_LIMIT_ = 10;
    int base_trap_limit_;
    double trap_limit_scaling_factor_;
    double trap_limit_penalty_factor_;
    double progress_boost_factor_;
    double H_PROGRESS_THRESHOLD_;

    bool in_trap_mode_ = false;
    Eigen::Vector3d trap_center_{0, 0, 0};
    double trap_radius_ = 0.0;
    RRTNode3DPtr trap_SI_override_ = nullptr;
    RRTNode3DPtr trap_GI_override_ = nullptr;
    int trap_mode_steps_remaining_ = 0;
    const int TRAP_MODE_MAX_STEPS_MULTIPLIER_ = 5;

    std::vector<std::pair<RRTNode3DPtr, RRTNode3DPtr>> trap_node_pairs_;

    void obstaclePercentageCallback(const std_msgs::Float64::ConstPtr &msg)
    {
      double percentage = msg->data;
      TRAP_COUNT_LIMIT_ = base_trap_limit_ + static_cast<int>(percentage * trap_limit_scaling_factor_);
      ROS_INFO("[BRRT_Simple_Case1] Obstacle percentage received: %.2f%%. Updated TRAP_COUNT_LIMIT to %d.",
               percentage, TRAP_COUNT_LIMIT_);
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

      // Reset trap detection variables
      h_past_ = DBL_MAX;
      trapCount_ = 0;
      trap_node_pairs_.clear();

      if (vis_ptr_)
      {
        vis_ptr_->visualize_balls(std::vector<visualization::BALL>{}, "trap/nodes", visualization::Color::red, 1.0);
        vis_ptr_->visualize_pairline(std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>{}, "trap/line", visualization::Color::red, 0.1);
      }
    }

    void logSamplingEvent(const std::string &type, const Eigen::Vector3d &x_rand,
                          const Eigen::Vector3d &x_new, const Eigen::Vector3d &nearest_node,
                          bool is_valid, double distance, const std::string &status)
    {
      // if (!sampling_log_stream_.is_open())
      //   return;

      json log_entry;
      log_entry["timestamp"] = ros::Time::now().toSec();
      log_entry["iteration"] = number_of_iterations_;
      log_entry["event_type"] = "sampling_attempt";
      log_entry["sampling_type"] = type; // "biased" or "uniform"
      log_entry["sample_target"] = {x_rand[0], x_rand[1], x_rand[2]};
      log_entry["sample_result_steer"] = {x_new[0], x_new[1], x_new[2]};
      log_entry["nearest_node"] = {nearest_node[0], nearest_node[1], nearest_node[2]};
      log_entry["distance"] = distance;
      log_entry["status"] = status;     // "normal" or "swap_trees"
      log_entry["is_valid"] = is_valid; // true = "normal", false = "invalid"
      // sampling_log_stream_ << log_entry.dump() << std::endl;
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

    void changeNodeParent(RRTNode3DPtr &node, RRTNode3DPtr &parent, const double &cost_from_parent)
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

    void update_cache_nearest_heuristic(RRTNode3DPtr nodeSi, kdtree *treeA, kdtree *treeB)
    {
      struct kdres *nodesB = kd_nearest_n(treeB, nodeSi->x.data(), 30);
      while (!kd_res_end(nodesB))
      {
        RRTNode3DPtr nodeGi = (RRTNode3DPtr)kd_res_item_data(nodesB);
        double h = computeH(nodeSi->x, nodeGi->x);
        cache.insert(nodeSi, treeA, nodeGi, treeB, h);
        kd_res_next(nodesB);
      }
      kd_res_free(nodesB);
    }

    void collectTreeNodes(const RRTNode3DPtr &root, std::vector<RRTNode3DPtr> &out_nodes)
    {
      out_nodes.clear();
      if (!root)
        return;
      std::queue<RRTNode3DPtr> Q;
      Q.push(root);
      while (!Q.empty())
      {
        RRTNode3DPtr n = Q.front();
        Q.pop();
        out_nodes.push_back(n);
        for (const auto &c : n->children)
          Q.push(c);
      }
    }

    void enterTrapMode(RRTNode3DPtr a, RRTNode3DPtr b)
    {
      if (in_trap_mode_)
        return;

      // center/radius covering the detected pair (small margin)
      trap_center_ = 0.5 * (a->x + b->x);
      trap_radius_ = std::max(1.0, (a->x - b->x).norm() / 2.0 + 0.6); // slightly larger margin to enclose obstacle region

      std::vector<RRTNode3DPtr> nodesA, nodesB;
      collectTreeNodes(start_node_, nodesA);
      collectTreeNodes(goal_node_, nodesB);

      // Prefer guides that maximize the heuristic (move search away from trap toward promising directions)
      double bestH = -DBL_MAX;
      trap_SI_override_ = nullptr;
      for (auto &n : nodesA)
      {
        double h = computeH(n->x, goal_node_->x); // heuristic of this start-tree node relative to goal
        if (h > bestH)
        {
          bestH = h;
          trap_SI_override_ = n;
        }
      }

      bestH = -DBL_MAX;
      trap_GI_override_ = nullptr;
      for (auto &n : nodesB)
      {
        double h = computeH(start_node_->x, n->x); // heuristic of this goal-tree node relative to start
        if (h > bestH)
        {
          bestH = h;
          trap_GI_override_ = n;
        }
      }

      // fallback if selection failed
      if (!trap_SI_override_)
        trap_SI_override_ = a;
      if (!trap_GI_override_)
        trap_GI_override_ = b;

      in_trap_mode_ = true;
      trap_mode_steps_remaining_ = std::max(20, TRAP_COUNT_LIMIT_ * TRAP_MODE_MAX_STEPS_MULTIPLIER_);

      ROS_WARN("[BRRT_Simple_Case1] Entered TRAP MODE center=(%.2f,%.2f,%.2f) radius=%.2f steps=%d",
               trap_center_[0], trap_center_[1], trap_center_[2], trap_radius_, trap_mode_steps_remaining_);
    }

    void exitTrapMode()
    {
      if (!in_trap_mode_)
        return;
      in_trap_mode_ = false;
      trap_SI_override_ = nullptr;
      trap_GI_override_ = nullptr;
      trap_node_pairs_.clear();
      trapCount_ = 0;
      trap_mode_steps_remaining_ = 0;
      ROS_WARN("[BRRT_Simple_Case1] Exited TRAP MODE.");
      if (vis_ptr_)
      {
        vis_ptr_->visualize_balls(std::vector<visualization::BALL>{}, "trap/nodes", visualization::Color::red, 1.0);
        vis_ptr_->visualize_pairline(std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>{}, "trap/line", visualization::Color::red, 0.1);
      }
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

    // --- OLD computePbias FUNCTION DELETED ---

    bool brrt_optimize(const Eigen::Vector3d &s, const Eigen::Vector3d &g)
    {
      ros::Time rrt_start_time = ros::Time::now();
      bool tree_connected = false;
      bool path_reverse = false;

      double h_start_goal = computeH(start_node_->x, goal_node_->x);
      h_past_ = h_start_goal; // Initialize h_past_ for trap detection

      /* kd tree init */
      kdtree *kdtree_1 = kd_create(3);
      kdtree *kdtree_2 = kd_create(3);
      kd_insert3(kdtree_1, start_node_->x[0], start_node_->x[1], start_node_->x[2], start_node_);
      kd_insert3(kdtree_2, goal_node_->x[0], goal_node_->x[1], goal_node_->x[2], goal_node_);
      RRTNode3DPtr selected_SI = start_node_, selected_GI = goal_node_;
      kdtree *treeA = kdtree_1;
      kdtree *treeB = kdtree_2;

      std::random_device rd;                                // Seed
      std::mt19937 gen(rd());                               // Mersenne Twister engine
      std::uniform_real_distribution<double> dis(0.0, 1.0); // Uniform distribution [0,1)

      /* main loop */
      number_of_iterations_ = 0;

      cache.insert(start_node_, treeA, goal_node_, treeB, h_start_goal); // insert start and goal node to cache

      for (number_of_iterations_ = 0; number_of_iterations_ < max_iteration_; ++number_of_iterations_)
      {
        /* random sampling */
        Eigen::Vector3d x_rand = get_sample_valid();
        Eigen::Vector3d x_new;
        double random01 = dis(gen);
        struct kdres *p_nearestA = nullptr, *p_nearestB = nullptr;
        RRTNode3DPtr nearest_nodeA, nearest_nodeB;

        // --- MODIFIED: pbias calculation logic ---
        double h_tmp = std::numeric_limits<double>::infinity();
        double pbias = brrt_optimize_p1_; // Default value
        double cur_h = 0.0;
        double h_distance = 0.0; // The drop in heuristic (h_past - h_tmp)

        // Try to get and remove the minimum heuristic pair between the two trees
        if (cache.popMinByTree(treeA, treeB, selected_SI, selected_GI, h_tmp))
        {

          // --- TRAP DETECTION ---
          h_distance = h_past_ - h_tmp; // Heuristic should decrease, so h_past > h_tmp
          if (h_distance < H_THRESHOLD_ && std::isfinite(h_distance) && std::isfinite(h_tmp))
          {
            trapCount_++;
            trap_node_pairs_.push_back({selected_SI, selected_GI});

            if (trapCount_ >= TRAP_COUNT_LIMIT_ && vis_ptr_)
            {
              ROS_WARN_THROTTLE(1.0, "TRAP STATE DETECTED! trapCount: %d", trapCount_);

              auto &trap_pair = trap_node_pairs_[TRAP_COUNT_LIMIT_ - 1]; // Get the 10th pair
              RRTNode3DPtr trap_node_A = trap_pair.first;
              RRTNode3DPtr trap_node_B = trap_pair.second;

              std::vector<visualization::BALL> trap_balls;
              visualization::BALL ball_A, ball_B;
              ball_A.radius = 0.3;
              ball_A.center = trap_node_A->x;
              ball_B.radius = 0.3;
              ball_B.center = trap_node_B->x;
              trap_balls.push_back(ball_A);
              trap_balls.push_back(ball_B);
              vis_ptr_->visualize_balls(trap_balls, "trap/nodes", visualization::Color::red, 1.0);

              std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> trap_line;
              trap_line.push_back({trap_node_A->x, trap_node_B->x});
              vis_ptr_->visualize_pairline(trap_line, "trap/line", visualization::Color::red, 0.1);
            }
          }
          else // This is either good progress or a reset
          {
            if (trapCount_ > 0)
            {
              if (vis_ptr_)
              {
                vis_ptr_->visualize_balls(std::vector<visualization::BALL>{}, "trap/nodes", visualization::Color::red, 1.0);
                vis_ptr_->visualize_pairline(std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>{}, "trap/line", visualization::Color::red, 0.1);
              }
            }
            trapCount_ = 0;
            trap_node_pairs_.clear();
          }
          h_past_ = h_tmp; // Update h_past_ for next iteration's check
          // --- END TRAP DETECTION ---

          // --- NEW: DYNAMIC pbias CALCULATION ---
          // Start with the base probability
          pbias = brrt_optimize_p1_;

          // 1. Apply Trap Penalty (if in a trap)
          if (trapCount_ > 0)
          {
            // Penalty increases linearly from 0 to trap_limit_penalty_factor_ as trapCount_ goes from 0 to TRAP_COUNT_LIMIT_
            double trap_penalty = trap_limit_penalty_factor_ * (static_cast<double>(trapCount_) / static_cast<double>(TRAP_COUNT_LIMIT_));
            pbias = pbias * (1.0 - trap_penalty);
          }
          // 2. Apply Progress Boost (if good progress and not in a trap)
          else if (h_distance > H_PROGRESS_THRESHOLD_)
          {
            // Boost increases with how significant the drop was, relative to the last heuristic
            double progress_boost_ratio = std::min(h_distance / h_past_, 1.0); // Don't boost more than 100%
            double progress_boost = progress_boost_factor_ * progress_boost_ratio;
            pbias = pbias + (1.0 - pbias) * progress_boost; // Asymptotically approach 1.0
          }

          // Clamp the probability between 0.0 and 1.0
          pbias = std::max(0.0, std::min(pbias, 1.0));
          // --- END NEW DYNAMIC pbias ---

          // After popping one, get the *next* lowest heuristic (if any left)
          double next_min_h = cache.getLowestHeuristicIfNotEmpty();

          if (std::isfinite(next_min_h))
          {
            cur_h = next_min_h;
          }
          else
          {
            cur_h = h_start_goal;
          }
        }
        else
        {
          // No entry in cache for this tree pair — fallback
          cur_h = h_start_goal;
          h_past_ = h_start_goal;    // Reset h_past_ if cache is empty
          pbias = brrt_optimize_p1_; // Use base probability
        }
        // --- END MODIFIED pbias logic ---

        if (random01 < pbias)
        {
          // This is now BIASED sampling
          nearest_nodeA = selected_SI;
          x_new = steer(nearest_nodeA->x, x_rand, steer_length_);

          bool is_valid = map_ptr_->isStateValid(x_new) && map_ptr_->isSegmentValid(nearest_nodeA->x, x_new);

          if (!is_valid)
          {
            logSamplingEvent("biased", x_rand, x_new, nearest_nodeA->x, is_valid, h_distance, "swap_trees");
            std::swap(treeA, treeB);
            path_reverse = !path_reverse;
            continue;
          }
          logSamplingEvent("biased", x_rand, x_new, nearest_nodeA->x, is_valid, h_distance, "normal");
          nearest_nodeB = selected_GI;
        }
        else
        {
          // This is now UNIFORM (RANDOM) sampling
          p_nearestA = kd_nearest3(treeA, x_rand[0], x_rand[1], x_rand[2]);

          if (p_nearestA == nullptr)
          {
            continue;
          }
          nearest_nodeA = (RRTNode3DPtr)kd_res_item_data(p_nearestA);
          kd_res_free(p_nearestA);
          x_new = steer(nearest_nodeA->x, x_rand, steer_length_);
          bool is_valid = map_ptr_->isStateValid(x_new) && map_ptr_->isSegmentValid(nearest_nodeA->x, x_new);

          if (!is_valid)
          {
            logSamplingEvent("uniform", x_rand, x_new, nearest_nodeA->x, is_valid, 0.0, "swap_trees");
            std::swap(treeA, treeB);
            path_reverse = !path_reverse;
            continue;
          }
          logSamplingEvent("uniform", x_rand, x_new, nearest_nodeA->x, is_valid, 0.0, "normal");

          p_nearestB = kd_nearest3(treeB, x_new[0], x_new[1], x_new[2]);
          if (p_nearestB == nullptr)
          {
            continue;
          }
          nearest_nodeB = (RRTNode3DPtr)kd_res_item_data(p_nearestB);
          kd_res_free(p_nearestB);
        }

        // Extend Node A by steer
        double dist_from_A = nearest_nodeA->cost_from_start + steer_length_;
        RRTNode3DPtr new_nodeA(nullptr);
        if (valid_tree_node_nums_ + 1 >= max_tree_node_nums_)
        {
          valid_tree_node_nums_ = max_tree_node_nums_; // max_node_num reached
          break;
        }
        new_nodeA = addTreeNode(nearest_nodeA, x_new, dist_from_A, steer_length_);

        kd_insert3(treeA, x_new[0], x_new[1], x_new[2], new_nodeA);
        update_cache_nearest_heuristic(new_nodeA, treeA, treeB); // update cache with new node

        /* request x_new's nearest node in treeB */
        /* Greedy steer & check connection */
        vector<Eigen::Vector3d> x_connects;
        bool isConnected = greedySteer(nearest_nodeB->x, x_new, x_connects, steer_length_);

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
          }
          update_cache_nearest_heuristic(new_nodeB, treeB, treeA);
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
          break;
        }
        else
        {
          std::swap(treeA, treeB);
          path_reverse = !path_reverse;
        }

        // --- ADDED FOR STEP-BY-STEP VISUALIZATION ---
        if (vis_ptr_ && step_delay_ > 0.0)
        {
          visualizeWholeTree();               // Show the current state of both trees
          ros::Duration(step_delay_).sleep(); // Pause for the specified duration
        }
        // --- END OF ADDED CODE ---

      } // End of sampling iteration
      final_path_use_time_ = (ros::Time::now() - rrt_start_time).toSec();
#ifdef DEBUG
      visualizeWholeTree();
#endif
      if (tree_connected)
      {
        final_path_ = path_list_.back();
      }
#ifdef DEBUG
      else if (valid_tree_node_nums_ == max_tree_node_nums_)
      {
        ROS_ERROR_STREAM("[BRRT_Optimize_case1]: NOT CONNECTED TO GOAL after " << max_tree_node_nums_ << " nodes added to rrt-tree");
      }
      else
      {
        ROS_ERROR_STREAM("[BRRT_Optimize_case1]: NOT CONNECTED TO GOAL after " << (ros::Time::now() - rrt_start_time).toSec() << " seconds");
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
      vis_ptr_->visualize_balls(tree_nodes, "case1/tree_vertice", visualization::Color::blue, 0.5);
      vis_ptr_->visualize_pairline(edges, "case1/tree_edges", visualization::Color::blue, 0.05);
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

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
#ifndef BRRT_SIMPLE_CASE3_H
#define BRRT_SIMPLE_CASE3_H

#include "occ_grid/occ_map.h"
#include "visualization/visualization.hpp"
#include "sampler.h"
#include "node.h"
#include "kdtree.h"

#include <ros/ros.h>
#include <utility>
#include <queue>
#include <algorithm>
#include <deque>
namespace path_plan
{
  class BRRT_Simple_Case3
  {
  public:
  BRRT_Simple_Case3() {};
  BRRT_Simple_Case3(const ros::NodeHandle &nh, const env::OccMap::Ptr &mapPtr) : nh_(nh), map_ptr_(mapPtr)
    {
      nh_.param("BRRT/steer_length", steer_length_, 0.0);
      nh_.param("BRRT/search_time", search_time_, 0.0);
      nh_.param("BRRT/max_tree_node_nums", max_tree_node_nums_, 0);

      // Fallback: if BRRT params missing or invalid, try the RRT namespace used by launch files
      {
        double tmp_d; int tmp_i;
        if (steer_length_ <= 0.0) {
          if (nh_.getParam("RRT/steer_length", tmp_d) && tmp_d > 0.0) {
            ROS_WARN_STREAM("[BRRT] BRRT/steer_length missing or invalid. Using fallback RRT/steer_length: " << tmp_d);
            steer_length_ = tmp_d;
          }
        }
        if (search_time_ <= 0.0) {
          if (nh_.getParam("RRT/search_time", tmp_d) && tmp_d > 0.0) {
            ROS_WARN_STREAM("[BRRT] BRRT/search_time missing or invalid. Using fallback RRT/search_time: " << tmp_d);
            search_time_ = tmp_d;
          }
        }
        if (max_tree_node_nums_ <= 2) {
          if (nh_.getParam("RRT/max_tree_node_nums", tmp_i) && tmp_i > 2) {
            ROS_WARN_STREAM("[BRRT] BRRT/max_tree_node_nums missing or invalid. Using fallback RRT/max_tree_node_nums: " << tmp_i);
            max_tree_node_nums_ = tmp_i;
          }
        }

        // Defensive defaults
        if (steer_length_ <= 0.0) { ROS_WARN_STREAM("[BRRT] invalid steer_length (<=0). Setting to default 1.0"); steer_length_ = 1.0; }
        if (search_time_ <= 0.0) { ROS_WARN_STREAM("[BRRT] invalid search_time (<=0). Setting to default 5.0"); search_time_ = 5.0; }
        if (max_tree_node_nums_ <= 2) { ROS_WARN_STREAM("[BRRT] invalid max_tree_node_nums (<=2). Setting to default 10000"); max_tree_node_nums_ = 10000; }
      }

      nh_.param("BRRT_Optimize/p1", brrt_optimize_p1_, 0.8);
      nh_.param("BRRT_Optimize/u_p", brrt_optimize_u_p, 2.0);
      nh_.param("BRRT_Optimize/step", brrt_optimize_step_, 0.1);

      nh_.param("BRRT_Optimize/alpha", brrt_optimize_alpha_, 0.5);
      nh_.param("BRRT_Optimize/beta", brrt_optimize_beta_, 0.3);
      nh_.param("BRRT_Optimize/gamma", brrt_optimize_gamma_, 0.5);
      nh_.param("BRRT_Optimize/max_iteration", max_iteration_, 0);
      nh_.param("BRRT_Optimize/enable2d", brrt_enable_2d, false);

      ROS_WARN_STREAM("[BRRT_Optimize_case3] param: steer_length: " << steer_length_);
      ROS_WARN_STREAM("[BRRT_Optimize_case3] param: search_time: " << search_time_);
      ROS_WARN_STREAM("[BRRT_Optimize_case3] param: max_tree_node_nums: " << max_tree_node_nums_);

      sampler_.setSamplingRange(mapPtr->getOrigin(), mapPtr->getMapSize());

      valid_tree_node_nums_ = 0;
      nodes_pool_.resize(max_tree_node_nums_);
      for (int i = 0; i < max_tree_node_nums_; ++i)
      {
        nodes_pool_[i] = new TreeNode;
      }
    }
    ~BRRT_Simple_Case3() {};

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
      brrt_optimize_alpha_ = 0.875;
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

    double rewire_radius_init_ = 5.0;
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
      // h = brrt_optimize_alpha_ * si_gi_dist + brrt_optimize_beta_ * si_G_dist + brrt_optimize_gamma_ * gi_S_dist;
      h = brrt_optimize_alpha_ * si_gi_dist + (1 - brrt_optimize_alpha_) * (si_G_dist + gi_S_dist);
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
      // std::cout << "[BRRT_Optimize_case3] randomPointInCircle: A: " << A.transpose() << " B: " << B.transpose() << std::endl;
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
    //-----------------------rewire--------------------------------------------
        double getAdaptiveRewireRadius(int tree_size)
    {
        if (tree_size <= 1) return rewire_radius_init_;

        double decay = std::log10(tree_size) / (double)tree_size;
        double r = rewire_radius_init_ * (1.0 + decay);
        
        return std::max(r, steer_length_ * 1.5); 
    }
    void rewire(RRTNode3DPtr &new_node, kdtree *tree_ptr)
    {
        int tree_size = kd_res_size(kd_nearest_range3(tree_ptr, new_node->x[0], new_node->x[1], new_node->x[2], DBL_MAX));
        double r_near = getAdaptiveRewireRadius(valid_tree_node_nums_);
        struct kdres *neighbors = kd_nearest_range3(tree_ptr, new_node->x[0], new_node->x[1], new_node->x[2], r_near);
        
        if (kd_res_size(neighbors) <= 1)
        {
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

        for (auto &nb : neighbor_nodes)
        {
            double dist = calDist(nb->x, new_node->x);
            double new_cost = nb->cost_from_start + dist;

            if (new_cost < new_node->cost_from_start)
            {
                if (map_ptr_->isSegmentValid(nb->x, new_node->x))
                {
                    changeNodeParent(new_node, nb, dist); 
                }
            }
        }

        for (auto &nb : neighbor_nodes)
        {
            double dist = calDist(new_node->x, nb->x);
            double new_cost = new_node->cost_from_start + dist;

            if (new_cost < nb->cost_from_start)
            {
                if (map_ptr_->isSegmentValid(new_node->x, nb->x))
                {
                    changeNodeParent(nb, new_node, dist);
                }
            }
        }
    }
    double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

    double rayCast3D(const Eigen::Vector3d &start, const Eigen::Vector3d &dir, double max_dist)
    {
        Eigen::Vector3d end = start + dir.normalized() * max_dist;
        RayCaster raycaster;
        bool need_ray = raycaster.setInput(start / map_ptr_->getResolution(), end / map_ptr_->getResolution());
        if (!need_ray) return max_dist;
        Eigen::Vector3d half = Eigen::Vector3d(0.5, 0.5, 0.5);
        Eigen::Vector3d ray_pt;
        if (!raycaster.step(ray_pt)) return max_dist;
        while (raycaster.step(ray_pt))
        {
            Eigen::Vector3d tmp = (ray_pt + half) * map_ptr_->getResolution();
            if (!map_ptr_->isStateValid(tmp))
            {
                return (tmp - start).norm();
            }
        }
        return max_dist;
    }

    Eigen::Vector3d AFBGSteer(const Eigen::Vector3d &x_near, const Eigen::Vector3d &x_rand, const Eigen::Vector3d &x_target, double steer_length_)
    {
        Eigen::Vector3d v_expand = (x_rand - x_near).normalized();
        
        using dispair = std::pair<double, Eigen::Vector3d>;
        struct DistCompare {
            bool operator()(const dispair& a, const dispair& b) {
                return a.first > b.first; 
            }
        };

        std::priority_queue<dispair, std::vector<dispair>, DistCompare> pqueue;

        double min_obs_dist = DBL_MAX;
        Eigen::Vector3d obs_vec(0,0,0);
        
        int num_rays = 32; // Number of rays for 3D sphere scanning
        double lidar_radius = 30.0; // Reduced to 5.0 for straighter paths (less conservative obstacle avoidance)

        // 3D raycasting using Fibonacci sphere
        for(int i = 0; i < num_rays; ++i) {
            double phi = acos(1.0 - 2.0 * (i + 0.5) / num_rays);
            double theta = M_PI * (1.0 + sqrt(5.0)) * (i + 0.5);
            Eigen::Vector3d dir(sin(phi)*cos(theta), sin(phi)*sin(theta), cos(phi));
            
            double d = rayCast3D(x_near, dir, lidar_radius);
            if(d < lidar_radius) {
              Eigen::Vector3d vec = dir * d;
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
        
        double phi_val = steer_length_ * sigmoid((dist_to_target / max_dist) * 5.0); // Increased multiplier for stronger target attraction 
        Eigen::Vector3d v_target = (x_target - x_near).normalized();

        double eta = 0.0;
        Eigen::Vector3d v_tangent(0,0,0);

        if (min_obs_dist < 2.0 * steer_length_) {
             eta = steer_length_ * sigmoid((min_obs_dist / (2.0 * steer_length_)) * 5.0);
             
             Eigen::Vector3d v_obs_dir = obs_vec.normalized(); 
             
             // 3D tangent logic: project v_expand onto the plane orthogonal to v_obs_dir
             Eigen::Vector3d proj = v_expand - v_expand.dot(v_obs_dir) * v_obs_dir;
             if (proj.norm() > 1e-6) {
                 v_tangent = proj.normalized();
             } else {
                 // Fallback if v_expand is parallel to v_obs_dir
                 Eigen::Vector3d arbitrary(1, 0, 0);
                 if (std::abs(v_obs_dir.x()) > 0.9) arbitrary = Eigen::Vector3d(0, 1, 0);
                 v_tangent = v_obs_dir.cross(arbitrary).normalized();
             }
             
             total_vec = v_expand + phi_val * v_target + eta * v_tangent;
        } 
        else {
             total_vec = v_expand + phi_val * v_target;
        }
        
        double step_size = std::min(steer_length_, dist_to_target);
        return x_near + total_vec.normalized() * step_size;
    }
    //-------------------------------------------------------------------------
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
      std::cout << "[BRRT_Optimize_case3] Start sampling..." << std::endl;
#endif
      cache.insert(start_node_, treeA, goal_node_, treeB, h_start_goal); // insert start and goal node to cache

      // Adaptive bias params
      int window_W = 30;
      double rho_min = 0.05, rho_max = 0.90;
      double eps_min = 1e-4, eps_max = 1e-3;
      double L_min = 3, L_max = 20;
      double mu_min = 2, mu_max = 8;
      double eta = 1e-6;

      std::deque<bool> trapped_window;
      std::deque<double> h_star_history;
      double h_star = h_start_goal;
      h_star_history.push_back(h_star);
      int c_k = 0;
      for (number_of_iterations_ = 0; number_of_iterations_ < 150000; ++number_of_iterations_)
      {
        /* random sampling */
        
        Eigen::Vector3d x_new;
        double random01 = dis(gen);
        struct kdres *p_nearestA = nullptr, *p_nearestB = nullptr;
        RRTNode3DPtr nearest_nodeA, nearest_nodeB;
        double h_tmp;
        bool has_heuristic = cache.getMinByTree(treeA, treeB, selected_SI, selected_GI, h_tmp);
        
        // 1. Calculate h_k
        double h_k = has_heuristic ? h_tmp : h_start_goal;
        
        // 2. Update h_k^*
        h_star = std::min(h_star, h_k);
        h_star_history.push_back(h_star);
        if (h_star_history.size() > window_W + 1) {
            h_star_history.pop_front();
        }
        
        // 3. Compute phi_k
        double phi_k = 0.0;
        if (!trapped_window.empty()) {
            int trapped_count = 0;
            for (bool trapped : trapped_window) {
                if (trapped) trapped_count++;
            }
            phi_k = (double)trapped_count / trapped_window.size();
        }
        
        // 4. Calculate r_k and bar_r_k
        double h_star_kW = h_star_history.front();
        double r_k = (h_star_kW - h_star) / (window_W * h_start_goal + eta);
        double bar_r_k = window_W * r_k;
        
        // 5. Calculate thresholds
        double eps_stall = eps_min + (eps_max - eps_min) * phi_k;
        double L_phi = std::ceil(std::max(L_min, L_max * (1.0 - phi_k)));
        double mu_phi = mu_min + (mu_max - mu_min) * phi_k;
        
        // 6. Update c_k
        if (r_k < eps_stall) {
            c_k++;
        } else {
            c_k = 0;
        }
        
        // 7. Calculate rho_bias (pbias)
        // double pbias = computePbias(
        //     brrt_optimize_p1_,
        //     h_start_goal,
        //     selected_SI->x,
        //     selected_GI->x);
        double pbias = 0.0;
        if (c_k >= L_phi) {
            pbias = rho_min;
        } else {
            pbias = rho_min + (rho_max - rho_min) * (1.0 - std::exp(-mu_phi * bar_r_k));
        }
        pbias = std::max(rho_min, std::min(rho_max, pbias));
        
        if (random01 < pbias)
        {
          TreeNode* rootOther = path_reverse ? start_node_ : goal_node_;
          Eigen::Vector3d x_tmp = randomPointInCircle(selected_SI->x, selected_GI->x);
          nearest_nodeA = selected_SI;
          x_new = AFBGSteer(nearest_nodeA->x, x_tmp, rootOther->x, steer_length_);
          if ((!map_ptr_->isStateValid(x_new)) || (!map_ptr_->isSegmentValid(nearest_nodeA->x, x_new)))
          {
            trapped_window.push_back(true);
            if(trapped_window.size() > window_W) trapped_window.pop_front();
            std::swap(treeA, treeB);
            path_reverse = !path_reverse;
            continue;
          }

          nearest_nodeB = selected_GI;
#ifdef DEBUG
          vis_ptr_->visualize_a_ball(x_tmp, 0.5, "/brrt_optimize/x_tmp", visualization::Color::red);
#endif
        }
        else
        {
          Eigen::Vector3d x_rand = get_sample_valid();
// x_new = map_ptr_->getFreeNodeInLine(nearest_nodeA->x, x_rand, brrt_optimize_step_);

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
          TreeNode* rootOther = path_reverse ? start_node_ : goal_node_;
          x_new = AFBGSteer(nearest_nodeA->x, x_rand, rootOther->x, steer_length_);
          if ((!map_ptr_->isStateValid(x_new)) || (!map_ptr_->isSegmentValid(nearest_nodeA->x, x_new)))
          {
            trapped_window.push_back(true);
            if(trapped_window.size() > window_W) trapped_window.pop_front();
            std::swap(treeA, treeB);
            path_reverse = !path_reverse;
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

        trapped_window.push_back(false);
        if(trapped_window.size() > window_W) trapped_window.pop_front();

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
        //-----------------------rewire------------------------------
        // rewire(new_nodeA, treeA);
        //-----------------------------------------------------------
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
            //-----------------------rewire------------------------------
            // rewire(new_nodeB, treeB);
            //-----------------------------------------------------------
          }
          update_cache_nearest_heuristic(new_nodeB,treeB,treeA);
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
          std::cout << "[BRRT_Optimize_case3]**********Find path after " << number_of_iterations_ << " iterations" << std::endl;
#endif
          // Removed break to allow continuous optimization
          break;
        }
        
        // Always swap trees to maintain balance
        std::swap(treeA, treeB);
        path_reverse = !path_reverse;


#ifdef DEBUG
        // visualizeWholeTree();

        // vis_ptr_->visualize_a_ball(x_new, 0.5, "/brrt_optimize/x_new", visualization::Color::green);
        // vis_ptr_->visualize_a_ball(nearest_nodeA->x, 0.5, "/brrt_optimize/nearest_nodeA", visualization::Color::black);
        // vis_ptr_->visualize_a_ball(nearest_nodeB->x, 0.5, "/brrt_optimize/nearest_nodeB", visualization::Color::white);
        // usleep(500000); // Sleep for 0.1 seconds to visualize the tree growth
#endif

        /* Swap treeA&B */

      } // End of sampling iteration
#ifdef DEBUG
      visualizeWholeTree();
#endif
      final_path_use_time_ = (ros::Time::now() - rrt_start_time).toSec();
      if (tree_connected)
      {

#ifdef DEBUG
        ROS_INFO_STREAM("[BRRT_Optimize_case3]: find_path_use_time: " << solution_cost_time_pair_list_.front().second << ", length: " << solution_cost_time_pair_list_.front().first);
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
        ROS_ERROR_STREAM("[BRRT_Optimize_case3]: NOT CONNECTED TO GOAL after " << max_tree_node_nums_ << " nodes added to rrt-tree");
      }
      else
      {
        ROS_ERROR_STREAM("[BRRT_Optimize_case3]: NOT CONNECTED TO GOAL after " << (ros::Time::now() - rrt_start_time).toSec() << " seconds");
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

#ifndef SOF_FUNCTION_H
#define SOF_FUNCTION_H

#include "occ_grid/occ_map.h"
#include "node.h"
#include "kdtree.h"

#include <ros/ros.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <random>
#include <queue>
#include <utility>

// Helper struct to organize SOF parameters
struct SOFParams {
    double epsilon;
    double epsilon_floor;
    double gamma;
    double weight_grade;
    double lidar_radius;
    int n_blocks;
};

class SOF_Function {
public:
    SOF_Function(const env::OccMap::Ptr &mapPtr, const SOFParams &params) 
        : map_ptr_(mapPtr), p_(params) 
    {
        // Initialize random number generator
        std::random_device rd;
        gen_ = std::mt19937(rd());
        rand01_ = std::uniform_real_distribution<double>(0.0, 1.0);
    }

    ~SOF_Function() {}

    // -------------------------------------------------------------------------
    // Core Function 1: Artificial Field-Based Greedy Steering (AFBGSteer)
    // -------------------------------------------------------------------------
    Eigen::Vector3d AFBGSteer(const Eigen::Vector3d &x_near, const Eigen::Vector3d &x_rand, 
                              const Eigen::Vector3d &x_target, double steer_length)
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
        
        // Scan surroundings (simulated LiDAR)
        for(int i = 0; i < p_.n_blocks; ++i) {
            double ang = i * M_PI / 4.0; // Note: You might want to adjust this if n_blocks != 8
            double d = rayCast(x_near, ang);
            if(d < p_.lidar_radius) {
              Eigen::Vector3d vec = Eigen::Vector3d(cos(ang), sin(ang), 0) * d;
              pqueue.push({d, vec});
            }
        }

        if (!pqueue.empty()) {
            min_obs_dist = pqueue.top().first;
            obs_vec = pqueue.top().second;
        }

        Eigen::Vector3d total_vec(0,0,0);
        double dist_to_target = (x_near - x_target).norm();
        double max_dist = map_ptr_->getMapSize()(0);
        
        double phi = steer_length * sigmoid((dist_to_target / max_dist) * 5.0); 
        Eigen::Vector3d v_target = (x_target - x_near).normalized();

        double eta = 0.0;
        Eigen::Vector3d v_tangent(0,0,0);

        // Artificial Potential Field Logic
        if (min_obs_dist < 2.0 * steer_length) {
             eta = steer_length * sigmoid((min_obs_dist / (2.0 * steer_length)) * 5.0);
             
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

    // -------------------------------------------------------------------------
    // Core Function 2: Weight Sampling based on open sectors
    // -------------------------------------------------------------------------
    Eigen::Vector3d weightSample(TreeNode* root_node, const Eigen::Vector3d& target_point, 
                                 kdtree* tree, bool rand_sampling) 
    {
        // Decay exploration rate
        p_.epsilon = std::max(p_.epsilon * p_.gamma, p_.epsilon_floor);
        
        // 20% chance to return target directly (Goal Bias)
        if (rand01_(gen_) < 0.2) return target_point;

        TreeNode* chosen_node = nullptr;

        if (rand_sampling) {
            // Find nearest node in tree to the target point to sample around it
            struct kdres *p_nearest = kd_nearest3(tree, target_point[0], target_point[1], target_point[2]);
            if (p_nearest) {
                chosen_node = (TreeNode*)kd_res_item_data(p_nearest);
                kd_res_free(p_nearest);
            } else {
                chosen_node = root_node;
            }
        }
        else {
            chosen_node = root_node;
        }

        struct Sector { double min_a, max_a, r, w; };
        std::vector<Sector> blocks;
        double angle_step = 2.0 * M_PI / p_.n_blocks;
        double sum_weight = 0.0;

        // Calculate weight for each sector based on obstacle distance
        for (int i = 0; i < p_.n_blocks; ++i) {
            Sector s;
            s.min_a = i * angle_step;
            s.max_a = (i + 1) * angle_step;
            double center_angle = s.min_a + angle_step / 2.0;
            s.r = rayCast(chosen_node->x, center_angle);
            
            // Weight is proportional to the distance to obstacle
            s.w = std::pow(s.r, p_.weight_grade);
            sum_weight += s.w;
            blocks.push_back(s);
        }

        // Weighted random selection of a sector
        double rand_val = rand01_(gen_) * sum_weight;
        double cur_sum = 0.0;
        Sector selected = blocks.back();
        
        for (const auto& b : blocks) {
            cur_sum += b.w;
            if (rand_val <= cur_sum) { selected = b; break; }
        }

        // Generate point within selected sector
        double r = rand01_(gen_) * selected.r;
        double theta = selected.min_a + rand01_(gen_) * (selected.max_a - selected.min_a);
        
        return Eigen::Vector3d(chosen_node->x[0] + r*cos(theta), chosen_node->x[1] + r*sin(theta), 0.0);
    } 
/*
    Eigen::Vector3d weightSample(TreeNode* root_node, const Eigen::Vector3d& target_point, kdtree* tree, bool rand_sampling) {
        epsilon_ = std::max(epsilon_ * gamma_, epsilon_floor_);
        
        if (rand01_(gen_) < 0.2) return target_point;

        TreeNode* chosen_node = nullptr;
        if (rand_sampling == true){
          // if (valid_tree_node_nums_ > 2 && rand01_(gen_) < epsilon_) {
          //    int idx = std::rand() % valid_tree_node_nums_;
          //    chosen_node = nodes_pool_[idx];
          // } 
          // else 
          {
              struct kdres *p_nearest = kd_nearest3(tree, target_point[0], target_point[1], target_point[2]);
              // if (p_nearest) 
              // {
                  chosen_node = (TreeNode*)kd_res_item_data(p_nearest);
                  kd_res_free(p_nearest);
              // } else {
              //     chosen_node = root_node;
              // }
          }
          // int idx = std::rand() % valid_tree_node_nums_;
          // chosen_node = nodes_pool_[idx];
        }
        else chosen_node = root_node;
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

        double rand_val = rand01_(gen_) * sum_weight;
        double cur_sum = 0.0;
        Sector selected = blocks.back();
        for (const auto& b : blocks) {
            cur_sum += b.w;
            if (rand_val <= cur_sum) { selected = b; break; }
        }

        // double r = std::sqrt(rand01_(gen_)) * selected.r;
        double r = rand01_(gen_) * selected.r;
        double theta = selected.min_a + rand01_(gen_) * (selected.max_a - selected.min_a);
        
        return Eigen::Vector3d(chosen_node->x[0] + r*cos(theta), chosen_node->x[1] + r*sin(theta), 0.0);
    } */
    // Reset epsilon logic if needed when planner resets
    void reset() {
        p_.epsilon = 0.9; 
    }

private:
    // Helper: Sigmoid function for smooth steering transitions
    double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
    
    // Helper: RayCast to measure distance to obstacles
    double rayCast(const Eigen::Vector3d &start, double angle) {
        Eigen::Vector3d dir(cos(angle), sin(angle), 0.0);
        double dist = 0.0;
        double step = map_ptr_->getResolution(); // Ensure map resolution is accessible
        Eigen::Vector3d current = start;
        
        while (dist < p_.lidar_radius) {
            current = start + dir * dist;
            if (!map_ptr_->isStateValid(current)) return dist;
            dist += step;
        }
        return p_.lidar_radius;
    }

    env::OccMap::Ptr map_ptr_;
    SOFParams p_;
    std::mt19937 gen_;
    std::uniform_real_distribution<double> rand01_;
};

#endif // SOF_FUNCTION_H
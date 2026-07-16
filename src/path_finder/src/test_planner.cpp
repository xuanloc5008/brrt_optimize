/*
Copyright (C) 2022 Hongkai Ye (kyle_yeh@163.com)
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
#include "self_msgs_and_srvs/GlbObsRcv.h"
#include "occ_grid/occ_map.h"
#include "path_finder/rrt_sharp.h"
#include "path_finder/rrt_star.h"
#include "path_finder/rrt.h"
#include "path_finder/brrt.h"
#include "path_finder/bg_brrt.h"
#include "path_finder/brrt_optimize_case3.h"
#include "path_finder/testcase.h"
#include "visualization/visualization.hpp"
#include "path_finder/sampler.h"
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <utility>
#include <vector>

class TesterPathFinder
{
private:
    ros::NodeHandle nh_;
    ros::Subscriber goal_sub_;
    ros::Timer execution_timer_;
    ros::ServiceClient rcv_glb_obs_client_;

    env::OccMap::Ptr env_ptr_;
    std::shared_ptr<visualization::Visualization> vis_ptr_;
    shared_ptr<path_plan::BRRT> brrt_ptr_;
    shared_ptr<path_plan::BG_BRRT> bg_brrt_ptr_;
    shared_ptr<path_plan::BRRT_Simple_Case3> brrt_optimize_case3_ptr;
    Eigen::Vector3d start_, goal_;
    double start_z_, goal_z_;

    // Fixed start/goal configuration (for reproducible tests + RViz)
    bool use_fixed_start_goal_;
    Eigen::Vector3d fixed_start_, fixed_goal_;
    double min_start_goal_dist_;
    int number_test_times_;
    bool visualize_paths_;
    double hold_after_test_sec_;

    bool run_brrt_, run_bg_brrt_, run_brrt_case3_;
    // Implement for testing path planning algorithms
    BiasSampler sampler_;
    BRRTExperimentMultiAlgo *manager;
    std::string input_param_;
    std::string output_result_;

public:
    TesterPathFinder(const ros::NodeHandle &nh) : nh_(nh)
    {
        env_ptr_ = std::make_shared<env::OccMap>();
        env_ptr_->init(nh_);

        vis_ptr_ = std::make_shared<visualization::Visualization>(nh_);
        vis_ptr_->registe<visualization_msgs::Marker>("start");
        vis_ptr_->registe<visualization_msgs::Marker>("goal");
        vis_ptr_->registe<visualization_msgs::Marker>("start_goal_line");

        // --- BRRT (vanilla bidirectional RRT) ---
        brrt_ptr_ = std::make_shared<path_plan::BRRT>(nh_, env_ptr_);
        brrt_ptr_->setVisualizer(vis_ptr_);
        vis_ptr_->registe<nav_msgs::Path>("brrt_final_path");
        vis_ptr_->registe<sensor_msgs::PointCloud2>("brrt_final_wpts");

        // --- BG_BRRT (biased-goal BRRT) ---
        bg_brrt_ptr_ = std::make_shared<path_plan::BG_BRRT>(nh_, env_ptr_);
        bg_brrt_ptr_->setVisualizer(vis_ptr_);
        vis_ptr_->registe<nav_msgs::Path>("bg_brrt_final_path");
        vis_ptr_->registe<sensor_msgs::PointCloud2>("bg_brrt_final_wpts");

        // --- BRRT_Case3 (heuristic-cache optimized) ---
        brrt_optimize_case3_ptr = std::make_shared<path_plan::BRRT_Simple_Case3>(nh_, env_ptr_);
        brrt_optimize_case3_ptr->setVisualizer(vis_ptr_);
        vis_ptr_->registe<nav_msgs::Path>("brrt_case3_final_path");
        vis_ptr_->registe<sensor_msgs::PointCloud2>("brrt_case3_final_wpts");

        goal_sub_ = nh_.subscribe("/goal", 1, &TesterPathFinder::goalCallback, this);
        execution_timer_ = nh_.createTimer(ros::Duration(1), &TesterPathFinder::executionCallback, this);
        rcv_glb_obs_client_ = nh_.serviceClient<self_msgs_and_srvs::GlbObsRcv>("/pub_glb_obs");

        nh_.param("start_z", start_z_, 1.5);
        nh_.param("goal_z", goal_z_, 1.5);

        // Fixed start/goal — default ON so experiments are reproducible.
        // Set use_fixed_start_goal:=false to restore random free-space sampling.
        nh_.param("use_fixed_start_goal", use_fixed_start_goal_, true);
        double sx, sy, sz, gx, gy, gz;
        nh_.param("start_x", sx, -80.0);
        nh_.param("start_y", sy, -80.0);
        nh_.param("start_z", sz, start_z_);
        nh_.param("goal_x", gx, 80.0);
        nh_.param("goal_y", gy, 80.0);
        nh_.param("goal_z", gz, goal_z_);
        fixed_start_ = Eigen::Vector3d(sx, sy, sz);
        fixed_goal_  = Eigen::Vector3d(gx, gy, gz);
        start_ = fixed_start_;
        goal_  = fixed_goal_;
        start_z_ = sz;
        goal_z_  = gz;

        nh_.param("min_start_goal_dist", min_start_goal_dist_, 50.0);
        nh_.param("number_test_times", number_test_times_, 1);
        nh_.param("visualize_paths", visualize_paths_, true);
        nh_.param("hold_after_test_sec", hold_after_test_sec_, 15.0);

        nh_.param("run_brrt",       run_brrt_,       true);
        nh_.param("run_bg_brrt",    run_bg_brrt_,    true);
        // Launch uses run_brrt_optimize; also accept run_brrt_case3.
        run_brrt_case3_ = true;
        if (nh_.hasParam("run_brrt_case3"))
            nh_.param("run_brrt_case3", run_brrt_case3_, true);
        else
            nh_.param("run_brrt_optimize", run_brrt_case3_, true);

        nh_.param("input_param",   input_param_,   std::string("brrt_input.json"));
        nh_.param("output_result", output_result_, std::string("evaluation/result.json"));
        std::cout << "input_param: "   << input_param_   << std::endl;
        std::cout << "output_result: " << output_result_ << std::endl;
        ROS_INFO_STREAM("[Tester] use_fixed_start_goal=" << (use_fixed_start_goal_ ? "true" : "false")
                        << " min_dist=" << min_start_goal_dist_
                        << " trials=" << number_test_times_);
        ROS_INFO_STREAM("[Tester] fixed start=(" << fixed_start_.transpose()
                        << ") goal=(" << fixed_goal_.transpose()
                        << ") dist=" << (fixed_goal_ - fixed_start_).norm());
        manager = new BRRTExperimentMultiAlgo(
            input_param_,
            output_result_);
    }
    ~TesterPathFinder()
    {
        delete manager;
    };

    void publishStartGoalMarkers()
    {
        // Larger spheres so start/goal are easy to spot on large maps.
        vis_ptr_->visualize_a_ball(start_, 2.0, "start", visualization::Color::pink);
        vis_ptr_->visualize_a_ball(goal_, 2.0, "goal", visualization::Color::steelblue);
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> line;
        line.emplace_back(start_, goal_);
        vis_ptr_->visualize_pairline(line, "start_goal_line", visualization::Color::yellow, 0.4);
    }

    // If preferred is occupied, search nearby free cells (keeps config "as fixed as possible").
    bool snapToNearbyFree(Eigen::Vector3d &p, const char *name, double search_radius = 40.0)
    {
        if (env_ptr_->isStateValid(p))
            return true;

        ROS_WARN_STREAM("[Tester] " << name << " " << p.transpose()
                        << " occupied/OOB — snapping to nearby free space (r<=" << search_radius << ")");

        Eigen::Vector3d best = p;
        double best_d = 1e9;
        bool found = false;
        // Local disk search around preferred (x,y), keep z.
        for (int i = 0; i < 8000; ++i)
        {
            const double u = static_cast<double>(rand()) / RAND_MAX;
            const double v = static_cast<double>(rand()) / RAND_MAX;
            const double r = search_radius * std::sqrt(u);
            const double th = 2.0 * M_PI * v;
            Eigen::Vector3d cand(p[0] + r * std::cos(th), p[1] + r * std::sin(th), p[2]);
            if (!env_ptr_->isStateValid(cand))
                continue;
            const double d = (cand - p).norm();
            if (d < best_d)
            {
                best_d = d;
                best = cand;
                found = true;
                if (d < 2.0)
                    break;
            }
        }
        if (!found)
        {
            ROS_ERROR_STREAM("[Tester] cannot find free space near " << name << " " << p.transpose());
            return false;
        }
        p = best;
        ROS_WARN_STREAM("[Tester] " << name << " snapped -> " << p.transpose() << " (delta=" << best_d << ")");
        return true;
    }

    bool validateStartGoal(Eigen::Vector3d &s, Eigen::Vector3d &g)
    {
        if (!snapToNearbyFree(s, "start") || !snapToNearbyFree(g, "goal"))
            return false;

        const double dist = (s - g).norm();
        if (dist < min_start_goal_dist_)
        {
            ROS_ERROR("[Tester] start/goal too close: dist=%.2f < min_start_goal_dist=%.2f",
                      dist, min_start_goal_dist_);
            return false;
        }
        return true;
    }

    // Sample random free start/goal that are at least min_start_goal_dist_ apart.
    bool sampleFarStartGoal(Eigen::Vector3d &s, Eigen::Vector3d &g)
    {
        const int max_tries = 20000;
        for (int t = 0; t < max_tries; ++t)
        {
            s = get_sample_valid();
            g = get_sample_valid();
            if ((s - g).norm() >= min_start_goal_dist_)
                return true;
        }
        ROS_ERROR("[Tester] failed to sample start/goal with dist >= %.2f after %d tries",
                  min_start_goal_dist_, max_tries);
        return false;
    }

    void goalCallback(const geometry_msgs::PoseStamped::ConstPtr &goal_msg)
    {
        // Interactive RViz mode: previous goal becomes start; click sets new goal.
        // Fixed config is ignored here so manual testing still works.
        goal_[0] = goal_msg->pose.position.x;
        goal_[1] = goal_msg->pose.position.y;
        goal_[2] = goal_msg->pose.position.z;

        // If the goal was set using RViz's 2D Nav Goal, force the tunable flying altitude
        if (std::abs(goal_[2]) < 0.01) {
            goal_[2] = goal_z_;
        }

        ROS_INFO_STREAM("\n-----------------------------\ngoal rcved at " << goal_.transpose());
        publishStartGoalMarkers();

        if (run_brrt_)
        {
            ROS_WARN("Starting BRRT");
            bool brrt_res = brrt_ptr_->plan(start_, goal_);
            ROS_WARN("Finished BRRT");
            {
                int num_nodes = brrt_ptr_->get_valid_tree_node_nums();
                int num_iterations = brrt_ptr_->get_number_of_iteration();
                if (brrt_res)
                {
                    vector<Eigen::Vector3d> final_path = brrt_ptr_->getPath();
                    vis_ptr_->visualize_path(final_path, "brrt_final_path");
                    vis_ptr_->visualize_pointcloud(final_path, "brrt_final_wpts");
                    vector<std::pair<double, double>> slns = brrt_ptr_->getSolutions();
                    ROS_INFO_STREAM("[BRRT] final path len: " << slns.back().first);
                }
                ROS_INFO("[BRRT]       nodes: %d, iters: %d, %s", num_nodes, num_iterations, brrt_res ? "SUCCESS" : "FAILED");
            }
        }

        if (run_bg_brrt_)
        {
            ROS_WARN("Starting BG_BRRT");
            bool bg_brrt_res = bg_brrt_ptr_->plan(start_, goal_);
            ROS_WARN("Finished BG_BRRT");
            {
                int num_nodes = bg_brrt_ptr_->get_valid_tree_node_nums();
                int num_iterations = bg_brrt_ptr_->get_number_of_iteration();
                if (bg_brrt_res)
                {
                    vector<Eigen::Vector3d> final_path = bg_brrt_ptr_->getPath();
                    vis_ptr_->visualize_path(final_path, "bg_brrt_final_path");
                    vis_ptr_->visualize_pointcloud(final_path, "bg_brrt_final_wpts");
                    vector<std::pair<double, double>> slns = bg_brrt_ptr_->getSolutions();
                    ROS_INFO_STREAM("[BG_BRRT] final path len: " << slns.back().first);
                }
                ROS_INFO("[BG_BRRT]    nodes: %d, iters: %d, %s", num_nodes, num_iterations, bg_brrt_res ? "SUCCESS" : "FAILED");
            }
        }

        if (run_brrt_case3_)
        {
            bool brrt_optimize_case3_res = brrt_optimize_case3_ptr->plan(start_, goal_);
            {
                int num_nodes = brrt_optimize_case3_ptr->get_valid_tree_node_nums();
                int num_iterations = brrt_optimize_case3_ptr->get_number_of_iteration();
                if (brrt_optimize_case3_res)
                {
                    vector<Eigen::Vector3d> final_path = brrt_optimize_case3_ptr->getPath();
                    vis_ptr_->visualize_path(final_path, "brrt_case3_final_path");
                    vis_ptr_->visualize_pointcloud(final_path, "brrt_case3_final_wpts");
                    vector<std::pair<double, double>> slns = brrt_optimize_case3_ptr->getSolutions();
                    ROS_INFO_STREAM("[BRRT_Case3] final path len: " << slns.back().first);
                }
                ROS_INFO("[BRRT_Case3] nodes: %d, iters: %d, %s", num_nodes, num_iterations, brrt_optimize_case3_res ? "SUCCESS" : "FAILED");
            }
        }

        // Keep markers after planning (planners may redraw start/goal smaller).
        publishStartGoalMarkers();
        start_ = goal_;
    }
    void print_vector3d(std::string name, Eigen::Vector3d &p)
    {
        std::cout << name << " x: " << p[0] << " y: " << p[1] << " z: " << p[2] << std::endl;
    }
    Eigen::Vector3d get_sample_valid()
    {
        Eigen::Vector3d x_rand;
        sampler_.setSamplingRange(env_ptr_->getOrigin(), env_ptr_->getMapSize());
        sampler_.samplingOnce(x_rand);
        long int count = 0;
        while (!env_ptr_->isStateValid(x_rand))
        {
            sampler_.samplingOnce(x_rand);
            if (++count % 1000000 == 0) {
                ROS_WARN("get_sample_valid stuck! x_rand=(%f, %f, %f)", x_rand(0), x_rand(1), x_rand(2));
            }
        }
        return x_rand;
    }
    void executionCallback(const ros::TimerEvent &event)
    {

        if (!env_ptr_->mapValid())
        {
            ROS_INFO("no map rcved yet.");
            self_msgs_and_srvs::GlbObsRcv srv;
            if (!rcv_glb_obs_client_.call(srv))
                ROS_WARN("Failed to call service /pub_glb_obs");
        }
        else
        {
            execution_timer_.stop();
            ROS_WARN("Timer tick");
#ifndef DEBUG
            experiment_test();
#endif
        }
    };
    void experiment_test()
    {
        const auto &input = manager->get_input();

        ROS_INFO("Running Test %d  (use_fixed_start_goal=%s, trials=%d)",
                 input.trial, use_fixed_start_goal_ ? "true" : "false", number_test_times_);

        // Resolve start/goal once when fixed (same pair for every trial).
        if (use_fixed_start_goal_)
        {
            start_ = fixed_start_;
            goal_  = fixed_goal_;
            if (!validateStartGoal(start_, goal_))
            {
                ROS_ERROR("[Tester] Fixed start/goal invalid. "
                          "Adjust start_x/y/z goal_x/y/z or min_start_goal_dist, or set use_fixed_start_goal:=false.");
                ros::shutdown();
                return;
            }
            ROS_WARN_STREAM("[Tester] FIXED start=" << start_.transpose()
                            << " goal=" << goal_.transpose()
                            << " dist=" << (goal_ - start_).norm());
            // Give RViz a moment to subscribe, then publish markers.
            ros::Duration(0.5).sleep();
            for (int k = 0; k < 5; ++k)
            {
                publishStartGoalMarkers();
                ros::Duration(0.1).sleep();
            }
        }

        for (int i = 0; i < number_test_times_; ++i)
        {
            if (!use_fixed_start_goal_)
            {
                if (!sampleFarStartGoal(start_, goal_))
                {
                    ros::shutdown();
                    return;
                }
            }
            // Re-publish every trial so markers stay visible while planners run.
            publishStartGoalMarkers();

            print_vector3d("start", start_);
            print_vector3d("goal", goal_);
            ROS_INFO("[Tester] trial %d/%d  dist=%.2f", i + 1, number_test_times_,
                     (goal_ - start_).norm());
            std::map<std::string, AlgoResult> algo_outputs;

            // --- BRRT (vanilla bidirectional RRT) ---
            if (run_brrt_)
            {
                brrt_ptr_->set_test_param(input.epsilon);
                bool brrt_res = brrt_ptr_->plan(start_, goal_);
                {
                    int num_nodes = brrt_ptr_->get_valid_tree_node_nums();
                    int num_iterations = brrt_ptr_->get_number_of_iteration();
                    if (brrt_res)
                    {
                        vector<std::pair<double, double>> slns = brrt_ptr_->getSolutions();
                        algo_outputs["BRRT"] = {true, slns.back().second, slns.back().first, num_nodes, num_iterations, start_, goal_};
                        if (visualize_paths_)
                        {
                            vector<Eigen::Vector3d> final_path = brrt_ptr_->getPath();
                            vis_ptr_->visualize_path(final_path, "brrt_final_path");
                            vis_ptr_->visualize_pointcloud(final_path, "brrt_final_wpts");
                        }
                    }
                    else
                    {
                        algo_outputs["BRRT"] = {false, brrt_ptr_->get_final_path_use_time_(), DBL_MAX, num_nodes, num_iterations, start_, goal_};
                    }
                    ROS_INFO("[BRRT]       nodes: %d, iters: %d, %s", num_nodes, num_iterations, brrt_res ? "SUCCESS" : "FAILED");
                }
                publishStartGoalMarkers();
            }

            // --- BG_BRRT (biased-goal BRRT) ---
            if (run_bg_brrt_)
            {
                bg_brrt_ptr_->set_test_param(input.epsilon);
                bool bg_brrt_res = bg_brrt_ptr_->plan(start_, goal_);
                {
                    int num_nodes = bg_brrt_ptr_->get_valid_tree_node_nums();
                    int num_iterations = bg_brrt_ptr_->get_number_of_iteration();
                    if (bg_brrt_res)
                    {
                        vector<std::pair<double, double>> slns = bg_brrt_ptr_->getSolutions();
                        algo_outputs["BG_BRRT"] = {true, slns.back().second, slns.back().first, num_nodes, num_iterations, start_, goal_};
                        if (visualize_paths_)
                        {
                            vector<Eigen::Vector3d> final_path = bg_brrt_ptr_->getPath();
                            vis_ptr_->visualize_path(final_path, "bg_brrt_final_path");
                            vis_ptr_->visualize_pointcloud(final_path, "bg_brrt_final_wpts");
                        }
                    }
                    else
                    {
                        algo_outputs["BG_BRRT"] = {false, bg_brrt_ptr_->get_final_path_use_time_(), DBL_MAX, num_nodes, num_iterations, start_, goal_};
                    }
                    ROS_INFO("[BG_BRRT]    nodes: %d, iters: %d, %s", num_nodes, num_iterations, bg_brrt_res ? "SUCCESS" : "FAILED");
                }
                publishStartGoalMarkers();
            }

            // --- BRRT_Case3 (heuristic-cache optimized) ---
            if (run_brrt_case3_)
            {
                brrt_optimize_case3_ptr->set_heuristic_param(input.p1, input.u_p, input.alpha, input.beta, input.gamma, input.epsilon);
                bool brrt_optimize_case3_res = brrt_optimize_case3_ptr->plan(start_, goal_);
                {
                    int num_nodes = brrt_optimize_case3_ptr->get_valid_tree_node_nums();
                    int num_iterations = brrt_optimize_case3_ptr->get_number_of_iteration();
                    if (brrt_optimize_case3_res)
                    {
                        vector<std::pair<double, double>> slns = brrt_optimize_case3_ptr->getSolutions();
                        algo_outputs["BRRT_Case3"] = {true, slns.back().second, slns.back().first, num_nodes, num_iterations, start_, goal_};
                        if (visualize_paths_)
                        {
                            vector<Eigen::Vector3d> final_path = brrt_optimize_case3_ptr->getPath();
                            vis_ptr_->visualize_path(final_path, "brrt_case3_final_path");
                            vis_ptr_->visualize_pointcloud(final_path, "brrt_case3_final_wpts");
                        }
                    }
                    else
                    {
                        algo_outputs["BRRT_Case3"] = {false, brrt_optimize_case3_ptr->get_final_path_use_time_(), DBL_MAX, num_nodes, num_iterations, start_, goal_};
                    }
                    ROS_INFO("[BRRT_Case3] nodes: %d, iters: %d, %s", num_nodes, num_iterations, brrt_optimize_case3_res ? "SUCCESS" : "FAILED");
                }
                publishStartGoalMarkers();
            }

            manager->store_output_for_run(algo_outputs);
        }
        manager->save_json();
        ROS_INFO("Completed all runs. Holding RViz markers for %.1f s (hold_after_test_sec)...",
                 hold_after_test_sec_);
        // Keep republishing so late RViz subscribers still see start/goal/path.
        const ros::Time t_end = ros::Time::now() + ros::Duration(hold_after_test_sec_);
        while (ros::ok() && ros::Time::now() < t_end)
        {
            publishStartGoalMarkers();
            ros::Duration(0.5).sleep();
        }
        ros::shutdown();
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "test_path_finder_node");
    ros::NodeHandle nh("~");

    TesterPathFinder tester(nh);

    ros::AsyncSpinner spinner(0);
    spinner.start();
    ros::waitForShutdown();
    return 0;
}

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

#define NUMBER_TEST_TIMES 100
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

        goal_ = get_sample_valid();
        goal_sub_ = nh_.subscribe("/goal", 1, &TesterPathFinder::goalCallback, this);
        execution_timer_ = nh_.createTimer(ros::Duration(1), &TesterPathFinder::executionCallback, this);
        rcv_glb_obs_client_ = nh_.serviceClient<self_msgs_and_srvs::GlbObsRcv>("/pub_glb_obs");

        start_.setZero();

        nh_.param("start_z", start_z_, 5.0);
        nh_.param("goal_z", goal_z_, 5.0);

        start_[2] = start_z_; // Start at a tunable flying altitude

        nh_.param("run_brrt",       run_brrt_,       true);
        nh_.param("run_bg_brrt",    run_bg_brrt_,    true);
        nh_.param("run_brrt_case3", run_brrt_case3_, true);
        if (!run_brrt_case3_) {
            nh_.param("run_brrt_optimize", run_brrt_case3_, false);
        }

        nh_.param("input_param",   input_param_,   std::string("brrt_input.json"));
        nh_.param("output_result", output_result_, std::string("/tmp/brrt_result.json"));
        std::cout << "input_param: "   << input_param_   << std::endl;
        std::cout << "output_result: " << output_result_ << std::endl;
        manager = new BRRTExperimentMultiAlgo(
            input_param_,
            output_result_);
    }
    ~TesterPathFinder()
    {
        delete manager;
    };

    void goalCallback(const geometry_msgs::PoseStamped::ConstPtr &goal_msg)
    {
        goal_[0] = goal_msg->pose.position.x;
        goal_[1] = goal_msg->pose.position.y;
        goal_[2] = goal_msg->pose.position.z;
        
        // If the goal was set using RViz's 2D Nav Goal, force the tunable flying altitude
        if (std::abs(goal_[2]) < 0.01) {
            goal_[2] = goal_z_;
        }

        ROS_INFO_STREAM("\n-----------------------------\ngoal rcved at " << goal_.transpose());
        vis_ptr_->visualize_a_ball(start_, 0.3, "start", visualization::Color::pink);
        vis_ptr_->visualize_a_ball(goal_, 0.3, "goal", visualization::Color::steelblue);

        // BiasSampler sampler;
        // sampler.setSamplingRange(env_ptr_->getOrigin(), env_ptr_->getMapSize());
        // vector<Eigen::Vector3d> preserved_samples;
        // for (int i = 0; i < 5000; ++i)
        // {
        //     Eigen::Vector3d rand_sample;
        //     sampler.uniformSamplingOnce(rand_sample);
        //     preserved_samples.push_back(rand_sample);
        // }
        // rrt_ptr_->setPreserveSamples(preserved_samples);
        // rrt_star_ptr_->setPreserveSamples(preserved_samples);
        // rrt_sharp_ptr_->setPreserveSamples(preserved_samples);

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
        
        ROS_INFO("Running Test %d", input.trial);
        for (int i = 0; i < NUMBER_TEST_TIMES; ++i)
        {
            start_ = get_sample_valid();
            goal_ = get_sample_valid();
            double dist = (start_ - goal_).norm();
            while (dist < 5)
            {
                start_ = get_sample_valid();
                goal_ = get_sample_valid();
                dist = (start_ - goal_).norm();
            }
            print_vector3d("start", start_);
            print_vector3d("goal", goal_);
            std::map<std::string, AlgoResult> algo_outputs;

            // --- BRRT (vanilla bidirectional RRT) ---
            if (run_brrt_) {
                brrt_ptr_->set_test_param(input.epsilon);
                bool brrt_res = brrt_ptr_->plan(start_, goal_);
                {
                    int num_nodes = brrt_ptr_->get_valid_tree_node_nums();
                    int num_iterations = brrt_ptr_->get_number_of_iteration();
                    if (brrt_res)
                    {
                        vector<std::pair<double, double>> slns = brrt_ptr_->getSolutions();
                        algo_outputs["BRRT"] = {true, slns.back().second, slns.back().first, num_nodes, num_iterations, start_, goal_};
                    }
                    else
                    {
                        algo_outputs["BRRT"] = {false, brrt_ptr_->get_final_path_use_time_(), DBL_MAX, num_nodes, num_iterations, start_, goal_};
                    }
                    ROS_INFO("[BRRT]       nodes: %d, iters: %d, %s", num_nodes, num_iterations, brrt_res ? "SUCCESS" : "FAILED");
                }
            }

            // --- BG_BRRT (biased-goal BRRT) ---
            if (run_bg_brrt_) {
                bg_brrt_ptr_->set_test_param(input.epsilon);
                bool bg_brrt_res = bg_brrt_ptr_->plan(start_, goal_);
                {
                    int num_nodes = bg_brrt_ptr_->get_valid_tree_node_nums();
                    int num_iterations = bg_brrt_ptr_->get_number_of_iteration();
                    if (bg_brrt_res)
                    {
                        vector<std::pair<double, double>> slns = bg_brrt_ptr_->getSolutions();
                        algo_outputs["BG_BRRT"] = {true, slns.back().second, slns.back().first, num_nodes, num_iterations, start_, goal_};
                    }
                    else
                    {
                        algo_outputs["BG_BRRT"] = {false, bg_brrt_ptr_->get_final_path_use_time_(), DBL_MAX, num_nodes, num_iterations, start_, goal_};
                    }
                    ROS_INFO("[BG_BRRT]    nodes: %d, iters: %d, %s", num_nodes, num_iterations, bg_brrt_res ? "SUCCESS" : "FAILED");
                }
            }

            // --- BRRT_Case3 (heuristic-cache optimized) ---
            if (run_brrt_case3_) {
                brrt_optimize_case3_ptr->set_heuristic_param(input.p1, input.u_p, input.alpha, input.beta, input.gamma, input.epsilon);
                bool brrt_optimize_case3_res = brrt_optimize_case3_ptr->plan(start_, goal_);
                {
                    int num_nodes = brrt_optimize_case3_ptr->get_valid_tree_node_nums();
                    int num_iterations = brrt_optimize_case3_ptr->get_number_of_iteration();
                    if (brrt_optimize_case3_res)
                    {
                        vector<std::pair<double, double>> slns = brrt_optimize_case3_ptr->getSolutions();
                        algo_outputs["BRRT_Case3"] = {true, slns.back().second, slns.back().first, num_nodes, num_iterations, start_, goal_};
                    }
                    else
                    {
                        algo_outputs["BRRT_Case3"] = {false, brrt_optimize_case3_ptr->get_final_path_use_time_(), DBL_MAX, num_nodes, num_iterations, start_, goal_};
                    }
                    ROS_INFO("[BRRT_Case3] nodes: %d, iters: %d, %s", num_nodes, num_iterations, brrt_optimize_case3_res ? "SUCCESS" : "FAILED");
                }
            }

            manager->store_output_for_run(algo_outputs);
        }
        manager->save_json();
        ROS_INFO("Completed all runs.");
        ros::shutdown();
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "test_path_finder_node");
    ros::NodeHandle nh("~");

    TesterPathFinder tester(nh);

    // Use single-threaded spinner to avoid data races on planner objects when experiment + goal callbacks interleave
    ros::AsyncSpinner spinner(1);
    spinner.start();
    ros::waitForShutdown();
    return 0;
}

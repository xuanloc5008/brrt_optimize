#pragma once
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>

using json = nlohmann::json;

struct BRRTInput {
    int trial;
    std::string environment;
    double p1;
    double u_p;
    double alpha;
    double beta;
    double gamma;
    double epsilon;
    // --- 4 BIẾN MỚI THÊM VÀO ĐỂ TEST ---
    double weight_grade;
    double lidar_radius;
    int n_blocks;
    double steer_length;
};

struct AlgoResult {
    bool success;
    double search_time;
    double path_length;
    int node_count;
    int num_iterations;
    Eigen::Vector3d start;
    Eigen::Vector3d goal;
};

struct RunResult {
    int run;
    std::map<std::string, AlgoResult> algorithms;
};

class BRRTExperimentMultiAlgo {
public:
    BRRTExperimentMultiAlgo(const std::string& json_input_path, const std::string& json_output_path)
        : json_out_path_(json_output_path), run_index_(1) {
        load_from_json(json_input_path);
    }

    const BRRTInput& get_input() const { return input_; }

    void store_output_for_run(const std::map<std::string, AlgoResult>& algo_results) {
        RunResult run;
        run.run = run_index_;
        run.algorithms = algo_results;
        results_.push_back(run);
        run_index_++;
    }

    int current_run_index() const { return run_index_; }

    void save_json() const {
        json j;
        j["trial"] = input_.trial;
        j["environment"] = input_.environment;
        j["parameters"] = {
            {"p1", input_.p1},
            {"u_p", input_.u_p},
            {"alpha", input_.alpha},
            {"beta", input_.beta},
            {"gamma", input_.gamma},
            {"epsilon", input_.epsilon},
            // --- LƯU 4 BIẾN MỚI ---
            {"weight_grade", input_.weight_grade},
            {"lidar_radius", input_.lidar_radius},
            {"n_blocks", input_.n_blocks},
            {"steer_length", input_.steer_length}
        };

        json result_array = json::array();
        for (const auto& run : results_) {
            json algos;
            for (const auto& kv : run.algorithms) {
                const std::string& name = kv.first;
                const AlgoResult& r = kv.second;
                algos[name] = {
                    {"success", r.success},
                    {"search_time", r.search_time},
                    {"path_length", r.path_length},
                    {"node_count", r.node_count},
                    {"num_iterations", r.num_iterations},
                    {"start", {r.start[0], r.start[1], r.start[2]}},
                    {"goal", {r.goal[0], r.goal[1], r.goal[2]}}
                };
            }
            result_array.push_back({
                {"run", run.run},
                {"algorithms", algos}
            });
        }

        j["results"] = result_array;

        std::ofstream file(json_out_path_);
        std::cout << "Saving results to: " << json_out_path_ << std::endl;
        file << j.dump(4);
    }

private:
    BRRTInput input_;
    std::vector<RunResult> results_;
    std::string json_out_path_;
    int run_index_;

    void load_from_json(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "Failed to open input JSON file: " << path << std::endl;
            return;
        }

        json j;
        file >> j;
        input_.trial = j.at("trial").get<int>();
        input_.environment = j.at("environment").get<std::string>();
        input_.p1 = j.at("p1").get<double>();
        input_.u_p = j.at("u_p").get<double>();
        input_.alpha = j.at("alpha").get<double>();
        input_.beta = j.at("beta").get<double>();
        input_.gamma = j.at("gamma").get<double>();
        input_.epsilon = j.at("epsilon").get<double>();
        
        // --- ĐỌC 4 BIẾN MỚI ---
        input_.weight_grade = j.value("weight_grade", 3.0); 
        input_.lidar_radius = j.value("lidar_radius", 15.0);
        input_.n_blocks = j.value("n_blocks", 16);
        input_.steer_length = j.value("steer_length", 0.5);
    }
};
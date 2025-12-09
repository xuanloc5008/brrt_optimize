#pragma once
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <cstdlib>

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
        : run_index_(1) {
        load_from_json(json_input_path);
        setup_output_paths(json_output_path);
    }

    const BRRTInput& get_input() const {
        return input_;
    }

    void store_output_for_run(const std::map<std::string, AlgoResult>& algo_results) {
        RunResult run;
        run.run = run_index_;
        run.algorithms = algo_results;
        results_.push_back(run);
        append_run_jsonl(run);
        run_index_++;
    }

    int current_run_index() const {
        return run_index_;
    }

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
            {"epsilon", input_.epsilon}
        };

        json result_array = json::array();
        for (const auto& run : results_) {
            result_array.push_back(serialize_run(run));
        }

        j["results"] = result_array;

        std::ofstream file(json_out_path_);
        if (!file.is_open()) {
            std::cerr << "Failed to open output JSON file: " << json_out_path_ << std::endl;
            return;
        }
        std::cout << "Saving results to: " << json_out_path_ << std::endl;
        file << j.dump(4);
    }

private:
    BRRTInput input_;
    std::vector<RunResult> results_;
    std::string json_out_path_;
    std::string jsonl_out_path_;
    std::ofstream jsonl_stream_;
    int run_index_;

    void setup_output_paths(const std::string& base_path) {
        json_out_path_ = resolve_output_path(base_path, "run_results.json", ".json");
        jsonl_out_path_ = resolve_output_path(base_path, "run_results.jsonl", ".jsonl");
        ensure_parent_dir(json_out_path_);
        ensure_parent_dir(jsonl_out_path_);
        jsonl_stream_.open(jsonl_out_path_, std::ios::out | std::ios::app);
        if (!jsonl_stream_.is_open()) {
            std::cerr << "Failed to open per-run log file: " << jsonl_out_path_ << std::endl;
        } else {
            std::cout << "Logging per-run metrics to: " << jsonl_out_path_ << std::endl;
        }
    }

    static std::string resolve_output_path(const std::string& base_path, const std::string& default_filename, const std::string& preferred_ext) {
        if (base_path.empty()) {
            return default_filename;
        }

        char last_char = base_path.back();
        if (last_char == '/' || last_char == '\\') {
            return base_path + default_filename;
        }

        size_t slash_pos = base_path.find_last_of("/\\");
        size_t dot_pos = base_path.find_last_of('.');
        bool has_extension = (dot_pos != std::string::npos) && (slash_pos == std::string::npos || dot_pos > slash_pos);
        if (has_extension) {
            std::string ext = base_path.substr(dot_pos);
            if (ext == preferred_ext) {
                return base_path;
            } else {
                return base_path.substr(0, dot_pos) + preferred_ext;
            }
        }
        return base_path + "/" + default_filename;
    }

    static std::string parent_dir(const std::string& path) {
        size_t slash_pos = path.find_last_of("/\\");
        if (slash_pos == std::string::npos) {
            return {};
        }
        return path.substr(0, slash_pos);
    }

    static void ensure_parent_dir(const std::string& path) {
        std::string dir = parent_dir(path);
        if (dir.empty()) {
            return;
        }
        std::string cmd = "mkdir -p \"" + dir + "\"";
        std::system(cmd.c_str());
    }

    json serialize_run(const RunResult& run) const {
        json algos;
        for (const auto& entry : run.algorithms) {
            const std::string& name = entry.first;
            const AlgoResult& r = entry.second;
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
        return {
            {"run", run.run},
            {"algorithms", algos}
        };
    }

    void append_run_jsonl(const RunResult& run) {
        if (!jsonl_stream_.is_open()) {
            return;
        }
        jsonl_stream_ << serialize_run(run).dump() << std::endl;
        jsonl_stream_.flush();
    }

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
        // json_out_path_ =  json_out_path_  + input_.environment + ".json";
    }
};

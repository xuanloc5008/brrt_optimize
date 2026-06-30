/*
Copyright (C) 2021 Hongkai Ye (kyle_yeh@163.com)
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

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/impl/kdtree.hpp>

#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <Eigen/Eigen>
#include <math.h>
#include <random>

using namespace std;
using namespace Eigen;

ros::Publisher _all_map_pub;
ros::Subscriber _odom_sub;

int _obs_num, _cir_num;
double _x_size, _y_size, _z_size, _init_x, _init_y, _resolution, _sense_rate;
double _x_l, _x_h, _y_l, _y_h, _w_l, _w_h, _h_l, _h_h, _w_c_l, _w_c_h;

bool _has_map = false;

sensor_msgs::PointCloud2 globalMap_pcd;
pcl::PointCloud<pcl::PointXYZ> cloudMap;

pcl::search::KdTree<pcl::PointXYZ> kdtreeMap;
vector<int> pointIdxSearch;
vector<float> pointSquaredDistance;
// --- Environment: Grid of Random L-Shapes (Corner Maze) with Clear Center ---
void GenerateEnv_L_Shape_Grid()
{
    cloudMap.points.clear();
    pcl::PointXYZ pt;

    auto add_rect = [&](double cx, double cy, double w, double h) {
        double x_start = cx - w / 2.0;
        double y_start = cy - h / 2.0;
        for (double i = x_start; i < x_start + w; i += _resolution)
            for (double j = y_start; j < y_start + h; j += _resolution)
                for (double k = -1.0; k < _h_h; k += _resolution) {
                    pt.x = i; pt.y = j; pt.z = k;
                    cloudMap.points.push_back(pt);
                }
    };

    double cell_size = 12.0;
    double arm_length = 8.0;
    double thickness = 2.5;
    
    // --- Define the Center Void Area ---
    // Adjust these values to make the center gap larger or smaller
    double clear_zone_radius = 15.0; 

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist_rot(0, 3);

    for (double x = _x_l + cell_size/2; x < _x_h - cell_size/2; x += cell_size)
    {
        for (double y = _y_l + cell_size/2; y < _y_h - cell_size/2; y += cell_size)
        {
            // --- SKIP LOGIC FOR CENTER ---
            // Assuming the map center is at (0,0). 
            // If the center is elsewhere, use (x - center_x) and (y - center_y)
            if (std::abs(x) < clear_zone_radius && std::abs(y) < clear_zone_radius)
            {
                continue; // Skip this cell to keep the center empty
            }

            int rotation = dist_rot(gen); 
            double h_center_x, h_center_y, h_w, h_h;
            double v_center_x, v_center_y, v_w, v_h;
            double offset = (arm_length - thickness) / 2.0;

            if (rotation == 0) { // L (Bottom-Left)
                h_w = arm_length; h_h = thickness; h_center_x = x; h_center_y = y - offset;
                v_w = thickness; v_h = arm_length; v_center_x = x - offset; v_center_y = y;
            }
            else if (rotation == 1) { // L (Bottom-Right)
                h_w = arm_length; h_h = thickness; h_center_x = x; h_center_y = y - offset;
                v_w = thickness; v_h = arm_length; v_center_x = x + offset; v_center_y = y;
            }
            else if (rotation == 2) { // L (Top-Right)
                h_w = arm_length; h_h = thickness; h_center_x = x; h_center_y = y + offset;
                v_w = thickness; v_h = arm_length; v_center_x = x + offset; v_center_y = y;
            }
            else { // L (Top-Left)
                h_w = arm_length; h_h = thickness; h_center_x = x; h_center_y = y + offset;
                v_w = thickness; v_h = arm_length; v_center_x = x - offset; v_center_y = y;
            }

            add_rect(h_center_x, h_center_y, h_w, h_h);
            add_rect(v_center_x, v_center_y, v_w, v_h);
        }
    }

    cloudMap.width = cloudMap.points.size();
    cloudMap.height = 1;
    cloudMap.is_dense = true;
    _has_map = true;
    pcl::toROSMsg(cloudMap, globalMap_pcd);
    globalMap_pcd.header.frame_id = "map";
    ROS_INFO("Generated L-Shape Grid map (Clear Center) with %zu points.", cloudMap.points.size());
}
void RandomBRRTGenerate_Large(double size = 4)
{
   pcl::PointXYZ pt_random;
   random_device rd;
   // float ramdom_ratio = 0.4; // Original
   float ramdom_ratio = 0.6;
   int number_ostacle = (_x_h - _x_l) * (_y_h - _y_l) / (size * size) * ramdom_ratio;
   std::cout << "number of ostacle" << number_ostacle;

   std::mt19937 gen(rd()); 

   std::uniform_real_distribution<> dis_x(_x_l, _x_h);
   std::uniform_real_distribution<> dis_y(_y_l, _y_h);
   
   // --- Define the Center Void Area ---
   // Obstacles will not be spawned if their center is within this distance from (0,0)
   double clear_zone_radius = 2.0; 

   double half_size = size / 2;
   for (int i = 0; i < number_ostacle; i++)
   {
      double random_x = dis_x(gen);
      double random_y = dis_y(gen);

      // --- SKIP LOGIC FOR CENTER ---
      // Checks if the sampled obstacle center falls within the clear zone
      if (std::abs(random_x) < clear_zone_radius && std::abs(random_y) < clear_zone_radius)
      {
         continue; 
      }

      for (double i_x = random_x - half_size; i_x < random_x + half_size; i_x += 0.5)
      {
         for (double i_y = random_y - half_size; i_y < random_y + half_size; i_y += 0.5)
         {
            for (float k = -1; k < _h_h; k += 0.1)
            {
               pt_random.x = i_x;
               pt_random.y = i_y;
               pt_random.z = k+1.0;
               cloudMap.points.push_back(pt_random);
            }
         }
      }
   }

   cloudMap.width = cloudMap.points.size();
   cloudMap.height = 1;
   cloudMap.is_dense = true;
   std::cout << "cloudMap.points.size() = " << cloudMap.points.size() << std::endl;
   _has_map = true;

   pcl::toROSMsg(cloudMap, globalMap_pcd);
   globalMap_pcd.header.frame_id = "map";
}

void RandomBRRTGenerate_Buildings(double size = 4)
{
   pcl::PointXYZ pt_random;
   random_device rd;
   // float ramdom_ratio = 0.4; // Original
   float ramdom_ratio = 0.4;
   int number_ostacle = (_x_h - _x_l) * (_y_h - _y_l) / (size * size) * ramdom_ratio;
   std::cout << "number of ostacle" << number_ostacle;

   std::mt19937 gen(rd()); 

   std::uniform_real_distribution<> dis_x(_x_l, _x_h);
   std::uniform_real_distribution<> dis_y(_y_l, _y_h);
   std::uniform_real_distribution<> dis_h(-1.0, _h_h); // Random height between floor (-1.0) and ceiling (_h_h)
   
   // --- Define the Center Void Area ---
   // Obstacles will not be spawned if their center is within this distance from (0,0)
   double clear_zone_radius = 2.0; 

   double half_size = size / 2;
   for (int i = 0; i < number_ostacle; i++)
   {
      double random_x = dis_x(gen);
      double random_y = dis_y(gen);
      double random_height = dis_h(gen);

      // --- SKIP LOGIC FOR CENTER ---
      // Checks if the sampled obstacle center falls within the clear zone
      if (std::abs(random_x) < clear_zone_radius && std::abs(random_y) < clear_zone_radius)
      {
         continue; 
      }

      for (double i_x = random_x - half_size; i_x < random_x + half_size; i_x += 0.5)
      {
         for (double i_y = random_y - half_size; i_y < random_y + half_size; i_y += 0.5)
         {
            for (float k = -1; k < random_height; k += 0.5)
            {
               pt_random.x = i_x;
               pt_random.y = i_y;
               pt_random.z = k;
               cloudMap.points.push_back(pt_random);
            }
         }
      }
   }

   cloudMap.width = cloudMap.points.size();
   cloudMap.height = 1;
   cloudMap.is_dense = true;
   std::cout << "cloudMap.points.size() = " << cloudMap.points.size() << std::endl;
   _has_map = true;

   pcl::toROSMsg(cloudMap, globalMap_pcd);
   globalMap_pcd.header.frame_id = "map";
}

void RandomBRRTGenerate_Cubes(double size = 4)
{
   pcl::PointXYZ pt_random;
   random_device rd;
   float ramdom_ratio = 0.4;
   double z_min = -1.0;
   
   // Adjust obstacle count for 3D volume
   int number_ostacle = (_x_h - _x_l) * (_y_h - _y_l) * (_h_h - z_min) / (size * size * size) * ramdom_ratio;
   std::cout << "number of ostacle: " << number_ostacle << std::endl;

   std::mt19937 gen(rd()); 

   std::uniform_real_distribution<> dis_x(_x_l, _x_h);
   std::uniform_real_distribution<> dis_y(_y_l, _y_h);
   std::uniform_real_distribution<> dis_z(z_min, _h_h);
   
   // --- Define the Center Void Area ---
   double clear_zone_radius = 2.0; 

   double half_size = size / 2;
   for (int i = 0; i < number_ostacle; i++)
   {
      double random_x = dis_x(gen);
      double random_y = dis_y(gen);
      double random_z = dis_z(gen);

      // --- SKIP LOGIC FOR CENTER ---
      if (std::abs(random_x) < clear_zone_radius && std::abs(random_y) < clear_zone_radius && std::abs(random_z) < clear_zone_radius)
      {
         continue; 
      }

      for (double i_x = random_x - half_size; i_x < random_x + half_size; i_x += 0.5)
      {
         for (double i_y = random_y - half_size; i_y < random_y + half_size; i_y += 0.5)
         {
            for (double i_z = random_z - half_size; i_z < random_z + half_size; i_z += 0.5)
            {
               pt_random.x = i_x;
               pt_random.y = i_y;
               pt_random.z = i_z;
               cloudMap.points.push_back(pt_random);
            }
         }
      }
   }

   cloudMap.width = cloudMap.points.size();
   cloudMap.height = 1;
   cloudMap.is_dense = true;
   std::cout << "cloudMap.points.size() = " << cloudMap.points.size() << std::endl;
   _has_map = true;

   pcl::toROSMsg(cloudMap, globalMap_pcd);
   globalMap_pcd.header.frame_id = "map";
}

void RobotArmMapGenerate()
{
   ROS_INFO("Generating Robot Arm Testing Map");
   cloudMap.points.clear();
   pcl::PointXYZ pt;

   auto add_box = [&](double x_min, double x_max, double y_min, double y_max, double z_min, double z_max) {
      for (double x = x_min; x <= x_max; x += _resolution) {
         for (double y = y_min; y <= y_max; y += _resolution) {
            for (double z = z_min; z <= z_max; z += _resolution) {
               pt.x = x; pt.y = y; pt.z = z;
               cloudMap.points.push_back(pt);
            }
         }
      }
   };

   auto add_table = [&](double cx, double cy, double cz, double w, double l, double h) {
      double t_thickness = 0.2;
      double leg_w = 0.2;
      add_box(cx - w/2, cx + w/2, cy - l/2, cy + l/2, cz + h - t_thickness, cz + h); // tabletop
      add_box(cx - w/2, cx - w/2 + leg_w, cy - l/2, cy - l/2 + leg_w, cz, cz + h - t_thickness); // leg 1
      add_box(cx + w/2 - leg_w, cx + w/2, cy - l/2, cy - l/2 + leg_w, cz, cz + h - t_thickness); // leg 2
      add_box(cx - w/2, cx - w/2 + leg_w, cy + l/2 - leg_w, cy + l/2, cz, cz + h - t_thickness); // leg 3
      add_box(cx + w/2 - leg_w, cx + w/2, cy + l/2 - leg_w, cy + l/2, cz, cz + h - t_thickness); // leg 4
   };

   auto add_chair = [&](double cx, double cy, double cz, double w, double l, double h) {
      double t_thickness = 0.2;
      double leg_w = 0.2;
      add_box(cx - w/2, cx + w/2, cy - l/2, cy + l/2, cz + h/2 - t_thickness, cz + h/2); // seat
      add_box(cx - w/2, cx - w/2 + leg_w, cy - l/2, cy - l/2 + leg_w, cz, cz + h/2 - t_thickness); // leg 1
      add_box(cx + w/2 - leg_w, cx + w/2, cy - l/2, cy - l/2 + leg_w, cz, cz + h/2 - t_thickness); // leg 2
      add_box(cx - w/2, cx - w/2 + leg_w, cy + l/2 - leg_w, cy + l/2, cz, cz + h/2 - t_thickness); // leg 3
      add_box(cx + w/2 - leg_w, cx + w/2, cy + l/2 - leg_w, cy + l/2, cz, cz + h/2 - t_thickness); // leg 4
      add_box(cx - w/2, cx - w/2 + leg_w, cy - l/2, cy + l/2, cz + h/2, cz + h); // backrest
   };

   auto add_cave = [&](double cx, double cy, double cz, double w, double l, double h) {
      double wall_t = 1.0;
      add_box(cx - w/2, cx - w/2 + wall_t, cy - l/2, cy + l/2, cz, cz + h); // left wall
      add_box(cx + w/2 - wall_t, cx + w/2, cy - l/2, cy + l/2, cz, cz + h); // right wall
      add_box(cx - w/2, cx + w/2, cy - l/2, cy + l/2, cz + h - wall_t, cz + h); // roof
   };

   auto add_flying_obj = [&](double cx, double cy, double cz, double w, double l, double h) {
      add_box(cx - w, cx + w, cy - l, cy + l, cz, cz + h); // floating box
   };

   auto add_wall_with_hole = [&](double cx, double cy, double cz, double w, double l, double h, double hole_w, double hole_h) {
      double wall_t = w;
      add_box(cx - wall_t/2, cx + wall_t/2, cy - l/2, cy + l/2, cz, cz + (h - hole_h)/2); // bottom wall
      add_box(cx - wall_t/2, cx + wall_t/2, cy - l/2, cy + l/2, cz + (h + hole_h)/2, cz + h); // top wall
      add_box(cx - wall_t/2, cx + wall_t/2, cy - l/2, cy - hole_w/2, cz + (h - hole_h)/2, cz + (h + hole_h)/2); // left side
      add_box(cx - wall_t/2, cx + wall_t/2, cy + hole_w/2, cy + l/2, cz + (h - hole_h)/2, cz + (h + hole_h)/2); // right side
   };

   // --- Original Robot Arm Map Elements ---
   // 1. Table (The floor/surface the robot is mounted on, slightly below 0 so the base is free)
   add_box(-20.0, 20.0, -20.0, 20.0, -1.0, -0.5);

   // 2. A "shelf" in front of the robot (to test reaching into/around)
   add_box(5.0, 8.0, -10.0, 10.0, 1.0, 1.5); // Shelf 1
   add_box(5.0, 8.0, -10.0, 10.0, 3.5, 4.0); // Shelf 2
   add_box(5.0, 8.0, -10.0, -9.0, 1.0, 8.0); // Left side panel
   add_box(5.0, 8.0,  9.0, 10.0, 1.0, 8.0);  // Right side panel

   // 3. A tall pillar blocking direct curved paths
   add_box(-2.0, 2.0, 4.0, 8.0, 0.0, 10.0);

   // 4. A ceiling obstacle (forces ducking or staying low)
   add_box(-10.0, 10.0, -10.0, 10.0, 8.0, 10.0);

   // 5. Some scattered blocks to simulate other equipment
   add_box(-10.0, -6.0, -10.0, -6.0, 0.0, 5.0);
   add_box(-8.0, -4.0,  6.0, 10.0, 0.0, 4.0);
   
   // 6. A "window" wall behind the robot (x = -15)
   add_box(-16.0, -14.0, -20.0, 20.0, 0.0, 2.0); // bottom wall
   add_box(-16.0, -14.0, -20.0, -5.0, 2.0, 8.0); // left wall
   add_box(-16.0, -14.0,  5.0,  20.0, 2.0, 8.0); // right wall

   // --- Random New Obstacles ---
   random_device rd;
   std::mt19937 gen(rd());
   int num_random_obs = _obs_num; // Uses obs_num parameter (controlled by obs_density in launch)

   std::uniform_real_distribution<> dis_x(_x_l, _x_h);
   std::uniform_real_distribution<> dis_y(_y_l, _y_h);
   std::uniform_real_distribution<> dis_z(1.0, _h_h - 4.0); // For floating objects
   std::uniform_int_distribution<> dis_type(0, 4); // 5 types of objects

   for (int i = 0; i < num_random_obs; ++i) {
      double cx = dis_x(gen);
      double cy = dis_y(gen);
      
      // Avoid spawning right around the robot base core (0,0)
      if (std::abs(cx) < 15.0 && std::abs(cy) < 15.0) {
         continue; 
      }

      int type = dis_type(gen);
      if (type == 0) {
         // Tall Pillar / Tree
         std::uniform_real_distribution<> height(5.0, 15.0);
         add_box(cx - 0.5, cx + 0.5, cy - 0.5, cy + 0.5, -0.5, height(gen));
      } else if (type == 1) {
         // Floating Window Gate
         double cz = dis_z(gen);
         add_wall_with_hole(cx, cy, cz, 1.0, 10.0, 8.0, 4.0, 4.0); 
      } else if (type == 2) {
         // Floating Tunnel / Cave
         double cz = dis_z(gen);
         add_cave(cx, cy, cz, 10.0, 10.0, 6.0);
      } else if (type == 3) {
         // Flying blocks
         std::uniform_real_distribution<> fly_z(1.0, _h_h - 2.0);
         add_flying_obj(cx, cy, fly_z(gen), 3.0, 3.0, 3.0);
      } else if (type == 4) {
         // Floating partial wall
         double cz = dis_z(gen);
         add_box(cx - 0.5, cx + 0.5, cy - 8.0, cy + 8.0, cz, cz + 8.0);
      }
   }

   cloudMap.width = cloudMap.points.size();
   cloudMap.height = 1;
   cloudMap.is_dense = true;
   std::cout << "Robot Arm map points size: " << cloudMap.points.size() << std::endl;
   _has_map = true;

   pcl::toROSMsg(cloudMap, globalMap_pcd);
   globalMap_pcd.header.frame_id = "map";
}

void RandomBRRTGenerate()
{
   random_device rd;
   default_random_engine eng(rd());
   float ramdom_ratio = 0.8;

   pcl::PointXYZ pt_random;
   std::cout<<"size of map" << _x_l << " " << _x_h << " " << _y_l << " " << _y_h <<" " << _h_h <<std::endl;
   for (float i = _x_l; i < _x_h; i+=0.5)
   {
     
      for (float j = _y_l; j < _y_h; j+=0.5)
      {
         // get a random number between 0 and 1
         float random_num = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
         if (random_num < ramdom_ratio)
         for (float k = -1; k < _h_h; k+=0.5)
         {
            pt_random.x = i;
            pt_random.y = j;
            pt_random.z = k;
            cloudMap.points.push_back(pt_random);
         }
      }
   }

   
   cloudMap.width = cloudMap.points.size();
   cloudMap.height = 1;
   cloudMap.is_dense = true;
   std::cout << "cloudMap.points.size() = " << cloudMap.points.size() << std::endl;
   _has_map = true;

   pcl::toROSMsg(cloudMap, globalMap_pcd);
   globalMap_pcd.header.frame_id = "map";
}
void addObstacleWall(double x1, double y1, double x2, double y2, double z_min, double z_max, double resolution)
{
   pcl::PointXYZ pt_obs;
   double dist = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
   int num_points_xy = dist / resolution;

   // Use num_points_xy + 1 to include both endpoints
   if (num_points_xy == 0) num_points_xy = 1; // Ensure at least one point if start/end are close

   double dx = (x2 - x1) / num_points_xy;
   double dy = (y2 - y1) / num_points_xy;

   for (double z = z_min; z <= z_max; z += resolution)
   {
      for (int i = 0; i <= num_points_xy; ++i)
      {
         pt_obs.x = x1 + i * dx;
         pt_obs.y = y1 + i * dy;
         pt_obs.z = z;
         cloudMap.points.push_back(pt_obs);
      }
   }
}

void FixedMapGenerate()
{
   ROS_INFO("Generating fixed map with outer L/H/U shapes and new central obstacles.");
   cloudMap.points.clear();

   double wall_z_min = -1.0;
   double wall_z_max = _h_h; // Use the parameter for height
   double res = _resolution; // Use the parameter for resolution

   // Ensure valid map boundaries
   if (_x_l >= _x_h || _y_l >= _y_h) {
      ROS_ERROR("Invalid map boundaries. Fixed map generation failed.");
      return;
   }

   // --- OUTER OBSTACLES (Unchanged) ---

   // 1. L-Shape (Bottom-Left)
   double l_base_x = _x_l + 5.0;
   double l_base_y = _y_l + 5.0;
   double l_len = 8.0;
   if (l_base_x + l_len < _x_h && l_base_y + l_len < _y_h) {
      addObstacleWall(l_base_x, l_base_y, l_base_x + l_len, l_base_y, wall_z_min, wall_z_max, res); // Horizontal part
      addObstacleWall(l_base_x, l_base_y, l_base_x, l_base_y + l_len, wall_z_min, wall_z_max, res); // Vertical part
   } else {
      ROS_WARN("L-Shape obstacle is outside map boundaries, skipping.");
   }

   // 2. H-Shape (Top-Right)
   double h_base_x = _x_h - 15.0;
   double h_base_y = _y_h - 15.0;
   double h_len = 8.0;
   double h_width = 6.0;
   if (h_base_x + h_width < _x_h && h_base_y + h_len < _y_h) {
      addObstacleWall(h_base_x, h_base_y, h_base_x, h_base_y + h_len, wall_z_min, wall_z_max, res); // Left bar
      addObstacleWall(h_base_x + h_width, h_base_y, h_base_x + h_width, h_base_y + h_len, wall_z_min, wall_z_max, res); // Right bar
      addObstacleWall(h_base_x, h_base_y + h_len / 2.0, h_base_x + h_width, h_base_y + h_len / 2.0, wall_z_min, wall_z_max, res); // Middle bar
   } else {
      ROS_WARN("H-Shape obstacle is outside map boundaries, skipping.");
   }

   // 3. U-Shape (Bottom-Right)
   double u_base_x = _x_h - 15.0;
   double u_base_y = _y_l + 5.0;
   double u_len = 8.0;
   double u_width = 6.0;
    if (u_base_x + u_width < _x_h && u_base_y + u_len < _y_h) {
      addObstacleWall(u_base_x, u_base_y, u_base_x, u_base_y + u_len, wall_z_min, wall_z_max, res); // Left bar
      addObstacleWall(u_base_x + u_width, u_base_y, u_base_x + u_width, u_base_y + u_len, wall_z_min, wall_z_max, res); // Right bar
      addObstacleWall(u_base_x, u_base_y, u_base_x + u_width, u_base_y, wall_z_min, wall_z_max, res); // Bottom bar
   } else {
      ROS_WARN("U-Shape obstacle is outside map boundaries, skipping.");
   }

   // 4. New H-Shape (Top-Left)
   double h2_base_x = _x_l + 5.0;
   double h2_base_y = _y_h - 15.0;
   double h2_len = 6.0;
   double h2_width = 5.0;
   if (h2_base_x + h2_width < _x_h && h2_base_y + h2_len < _y_h) {
      addObstacleWall(h2_base_x, h2_base_y, h2_base_x, h2_base_y + h2_len, wall_z_min, wall_z_max, res); // Left bar
      addObstacleWall(h2_base_x + h2_width, h2_base_y, h2_base_x + h2_width, h2_base_y + h2_len, wall_z_min, wall_z_max, res); // Right bar
      addObstacleWall(h2_base_x, h2_base_y + h2_len / 2.0, h2_base_x + h2_width, h2_base_y + h2_len / 2.0, wall_z_min, wall_z_max, res); // Middle bar
   } else {
      ROS_WARN("New H-Shape obstacle is outside map boundaries, skipping.");
   }

   // 5. New U-Shape (Inverted "n-shape", Top-Left)
   double u2_base_x = _x_l + 12.0; 
   double u2_base_y = _y_h - 10.0; 
   double u2_len = 6.0;
   double u2_width = 5.0;
   if (u2_base_x + u2_width < _x_h && u2_base_y + u2_len < _y_h) {
      addObstacleWall(u2_base_x, u2_base_y, u2_base_x, u2_base_y + u2_len, wall_z_min, wall_z_max, res); // Left bar
      addObstacleWall(u2_base_x + u2_width, u2_base_y, u2_base_x + u2_width, u2_base_y + u2_len, wall_z_min, wall_z_max, res); // Right bar
      addObstacleWall(u2_base_x, u2_base_y + u2_len, u2_base_x + u2_width, u2_base_y + u2_len, wall_z_min, wall_z_max, res); // Top bar
   } else {
       ROS_WARN("New U-Shape obstacle is outside map boundaries, skipping.");
   }

   // 6. New L-Shape (Inverted, near Bottom-Right)
   double l2_base_x = 5.0;
   double l2_base_y = -15.0;
   double l2_len = 6.0;
   if (l2_base_x - l2_len > _x_l && l2_base_y + l2_len < _y_h) {
       addObstacleWall(l2_base_x, l2_base_y, l2_base_x - l2_len, l2_base_y, wall_z_min, wall_z_max, res); // Horizontal bar (left)
       addObstacleWall(l2_base_x, l2_base_y, l2_base_x, l2_base_y + l2_len, wall_z_min, wall_z_max, res); // Vertical bar (up)
   } else {
       ROS_WARN("New L-Shape is outside map boundaries, skipping.");
   }

   // --- NEW: CENTRAL OBSTACLES ---

   // 7. Central Vertical Wall (Positive X)
   double v_wall_x = 2.0;
   double v_wall_y_start = -5.0;
   double v_wall_y_end = 5.0;
   if (v_wall_x > _x_l && v_wall_x < _x_h && v_wall_y_end < _y_h && v_wall_y_start > _y_l) {
      addObstacleWall(v_wall_x, v_wall_y_start, v_wall_x, v_wall_y_end, wall_z_min, wall_z_max, res);
   } else {
       ROS_WARN("Central Vertical Wall is outside map boundaries, skipping.");
   }
   
   // 8. Central Horizontal Wall (Positive Y)
   double h_wall_x_start = -5.0;
   double h_wall_x_end = 5.0;
   double h_wall_y = 2.0;
   if (h_wall_x_end < _x_h && h_wall_x_start > _x_l && h_wall_y < _y_h && h_wall_y > _y_l) {
      addObstacleWall(h_wall_x_start, h_wall_y, h_wall_x_end, h_wall_y, wall_z_min, wall_z_max, res);
   } else {
       ROS_WARN("Central Horizontal Wall is outside map boundaries, skipping.");
   }

   // 9. Small Box (Negative X, Negative Y)
   double box_x = -8.0;
   double box_y = -8.0;
   double box_size = 3.0;
   if (box_x + box_size < _x_h && box_y + box_size < _y_h) {
      addObstacleWall(box_x, box_y, box_x + box_size, box_y, wall_z_min, wall_z_max, res); // Bottom
      addObstacleWall(box_x + box_size, box_y, box_x + box_size, box_y + box_size, wall_z_min, wall_z_max, res); // Right
      addObstacleWall(box_x, box_y + box_size, box_x + box_size, box_y + box_size, wall_z_min, wall_z_max, res); // Top
   } else {
      ROS_WARN("Central Small Box is outside map boundaries, skipping.");
   }

   // --- End of Added Obstacles ---


   // Set cloud properties and convert to ROS message
   cloudMap.width = cloudMap.points.size();
   cloudMap.height = 1;
   cloudMap.is_dense = true;

   _has_map = true;
   ROS_INFO("Fixed map generated with %zu points.", cloudMap.points.size());

   pcl::toROSMsg(cloudMap, globalMap_pcd);
   globalMap_pcd.header.frame_id = "map";
}
void RandomNarrowGenerate()
{
   random_device rd;
   default_random_engine eng(rd());
   float t_y_l;
   float t_y_h;

   pcl::PointXYZ pt_random;
   int escape = 4;
   int step = 0;
   for (int i = int(_x_l); i < int(_x_h); i += escape)
   {
      if (step % 2 == 0)
      {
         t_y_l = _y_l;
         t_y_h = _y_h - escape;
      }
      else
      {
         t_y_l = _y_l + escape;
         t_y_h = _y_h;
      }
      step++;
      for (float j = t_y_l; j < t_y_h; j += 0.5)
      {
         for (float k = -1; k < _h_h; k += 0.5)
         {
            pt_random.x = i;
            pt_random.y = j;
            pt_random.z = k;
            cloudMap.points.push_back(pt_random);
         }
         // pt_random.x = i;
         // pt_random.y = j;
         // pt_random.z = -0.5;
         // cloudMap.points.push_back(pt_random);
      }
   }

   cloudMap.width = cloudMap.points.size();
   cloudMap.height = 1;
   cloudMap.is_dense = true;

   _has_map = true;

   pcl::toROSMsg(cloudMap, globalMap_pcd);
   globalMap_pcd.header.frame_id = "map";
}
// --- Environment A: Two Simple Obstacles ---
void GenerateEnvA_Simple()
{
   cloudMap.points.clear();
   pcl::PointXYZ pt;

   // Define two large square obstacles
   // Obstacle 1: Bottom-Left
   double obs1_x = -10.0, obs1_y = -10.0, size = 6.0;
   for (double i = obs1_x; i < obs1_x + size; i += _resolution)
      for (double j = obs1_y; j < obs1_y + size; j += _resolution)
         for (double k = -1; k < _h_h; k += _resolution) {
            pt.x = i; pt.y = j; pt.z = k;
            cloudMap.points.push_back(pt);
         }

   // Obstacle 2: Top-Right
   double obs2_x = 10.0, obs2_y = 10.0;
   for (double i = obs2_x; i < obs2_x + size; i += _resolution)
      for (double j = obs2_y; j < obs2_y + size; j += _resolution)
         for (double k = -1; k < _h_h; k += _resolution) {
            pt.x = i; pt.y = j; pt.z = k;
            cloudMap.points.push_back(pt);
         }

   cloudMap.width = cloudMap.points.size();
   cloudMap.height = 1;
   cloudMap.is_dense = true;
   _has_map = true;
   pcl::toROSMsg(cloudMap, globalMap_pcd);
   globalMap_pcd.header.frame_id = "map";
}

// --- Environment B: Cluttered Random Blocks ---
void GenerateEnvB_Cluttered()
{
   cloudMap.points.clear();
   pcl::PointXYZ pt;
   
   // Random generator
   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_real_distribution<> dis_x(_x_l + 2.0, _x_h - 2.0); // Padding from edges
   std::uniform_real_distribution<> dis_y(_y_l + 2.0, _y_h - 2.0);

   int num_obstacles = 40; // Number of small blocks
   double obs_size = 2.0;  // Size of each block

   for (int n = 0; n < num_obstacles; n++)
   {
      double cx = dis_x(gen);
      double cy = dis_y(gen);

      // Keep start (0,0) clear roughly
      if (std::sqrt(cx*cx + cy*cy) < 5.0) continue;

      for (double i = cx - obs_size/2; i < cx + obs_size/2; i += _resolution)
         for (double j = cy - obs_size/2; j < cy + obs_size/2; j += _resolution)
            for (double k = -1; k < _h_h; k += _resolution) {
               pt.x = i; pt.y = j; pt.z = k;
               cloudMap.points.push_back(pt);
            }
   }

   cloudMap.width = cloudMap.points.size();
   cloudMap.height = 1;
   cloudMap.is_dense = true;
   _has_map = true;
   pcl::toROSMsg(cloudMap, globalMap_pcd);
   globalMap_pcd.header.frame_id = "map";
}

// --- Environment C: Complex Maze / Trap ---
void GenerateEnvC_Maze()
{
   cloudMap.points.clear();
   pcl::PointXYZ pt;
   
   auto add_rect = [&](double x, double y, double w, double h) {
       for(double i=x; i<x+w; i+=_resolution)
           for(double j=y; j<y+h; j+=_resolution)
               for(double k=-1; k<_h_h; k+=_resolution) {
                   pt.x=i; pt.y=j; pt.z=k; cloudMap.points.push_back(pt);
               }
   };

   // 1. The "U" trap near the start (forces backtracking)
   // Left wall of U
   add_rect(5.0, -5.0, 2.0, 10.0);
   // Top wall of U
   add_rect(5.0, 5.0, 8.0, 2.0);
   // Bottom wall of U
   add_rect(5.0, -7.0, 8.0, 2.0);
   
   // 2. Some scattered maze walls similar to image C
   add_rect(-10.0, 5.0, 2.0, 10.0);  // Tall wall left
   add_rect(15.0, -10.0, 2.0, 15.0); // Tall wall right
   add_rect(-5.0, -10.0, 10.0, 2.0); // Horizontal bar
   add_rect(0.0, 12.0, 10.0, 2.0);   // Horizontal top

   cloudMap.width = cloudMap.points.size();
   cloudMap.height = 1;
   cloudMap.is_dense = true;
   _has_map = true;
   pcl::toROSMsg(cloudMap, globalMap_pcd);
   globalMap_pcd.header.frame_id = "map";
}

// --- Environment D: Narrow Passage ---
void GenerateEnvD_Narrow()
{
   cloudMap.points.clear();
   pcl::PointXYZ pt;

   double wall_x     = 0.0;    // wall centre x
   double wall_width = 70.0;   // wall thickness  → faces at x = ±35
   double gap_size   = 1.0;    // passage height (y: -0.5 … +0.5)

   double face_l = wall_x - wall_width / 2.0;   // left  face x = -35
   double face_r = wall_x + wall_width / 2.0;   // right face x = +35

   // ── Main wall (lower + upper blocks, gap at y = 0) ───────────────────
   for (double i = face_l; i < face_r; i += _resolution)
      for (double j = _y_l; j < -gap_size / 2; j += _resolution)
         for (double k = -1; k < _h_h; k += _resolution) {
            pt.x = i; pt.y = j; pt.z = k;
            cloudMap.points.push_back(pt);
         }

   for (double i = face_l; i < face_r; i += _resolution)
      for (double j = gap_size / 2; j < _y_h; j += _resolution)
         for (double k = -1; k < _h_h; k += _resolution) {
            pt.x = i; pt.y = j; pt.z = k;
            cloudMap.points.push_back(pt);
         }

   // ── Small obstacles at each tunnel head ───────────────────────────────
   // Placed just outside the wall faces, staggered above/below the gap
   // centre so the robot must approach at an angle — but a clear path
   // always exists beside each obstacle.
   double obs = 2.0;   // obstacle side length

   auto add_obs = [&](double cx, double cy) {
      for (double x = cx - obs/2; x < cx + obs/2; x += _resolution)
         for (double y = cy - obs/2; y < cy + obs/2; y += _resolution)
            for (double z = -1.0; z < _h_h; z += _resolution) {
               pt.x = x; pt.y = y; pt.z = z;
               cloudMap.points.push_back(pt);
            }
   };

   // Left entrance (approaching from x < -35)
   // — one obstacle above gap, close to face; one below gap, further out
   add_obs(face_l - 3.0,  2.5);    // above-centre, near wall
   add_obs(face_l - 7.0, -2.5);    // below-centre, further out

   // Right exit (leaving toward x > +35)
   // — mirrored: one below gap near face, one above gap further out
   add_obs(face_r + 3.0, -2.5);    // below-centre, near wall
   add_obs(face_r + 7.0,  2.5);    // above-centre, further out

   cloudMap.width = cloudMap.points.size();
   cloudMap.height = 1;
   cloudMap.is_dense = true;
   _has_map = true;
   pcl::toROSMsg(cloudMap, globalMap_pcd);
   globalMap_pcd.header.frame_id = "map";
}
void FixedTrapAndNarrowGenerate()
{
   pcl::PointXYZ pt_random;
   
   // Clear previous data if any
   cloudMap.points.clear();

   // 1. Create a Divider Wall with a Narrow Passage at X = 0
   // The gap is located at (0, 0) with a width of 1.2 meters
   double gap_width = 1.2; 
   double wall_thickness = 1.0;
   
   // Lower part of the wall
   for (double i = -wall_thickness/2; i < wall_thickness/2; i += _resolution) {
      for (double j = _y_l; j < -gap_width/2; j += _resolution) {
         for (double k = -1; k < _h_h; k += _resolution) {
            pt_random.x = i; pt_random.y = j; pt_random.z = k;
            cloudMap.points.push_back(pt_random);
         }
      }
   }
   // Upper part of the wall
   for (double i = -wall_thickness/2; i < wall_thickness/2; i += _resolution) {
      for (double j = gap_width/2; j < _y_h; j += _resolution) {
         for (double k = -1; k < _h_h; k += _resolution) {
            pt_random.x = i; pt_random.y = j; pt_random.z = k;
            cloudMap.points.push_back(pt_random);
         }
      }
   }

   // 2. Create a "U" Shaped Dead-End Trap at X = 10
   // It is open towards the start, forcing the robot to enter and backtrack
   double trap_center_x = 10.0;
   double trap_width = 8.0;
   double trap_depth = 6.0;
   
   // Back wall of the trap
   for (double j = -trap_width/2; j < trap_width/2; j += _resolution) {
      for (double i = trap_center_x + trap_depth; i < trap_center_x + trap_depth + wall_thickness; i += _resolution) {
          for (double k = -1; k < _h_h; k += _resolution) {
            pt_random.x = i; pt_random.y = j; pt_random.z = k;
            cloudMap.points.push_back(pt_random);
         }
      }
   }
   // Side walls of the trap
   for (double i = trap_center_x; i < trap_center_x + trap_depth; i += _resolution) {
      // Side 1
      for (double j = -trap_width/2 - wall_thickness; j < -trap_width/2; j += _resolution) {
         for (double k = -1; k < _h_h; k += _resolution) {
            pt_random.x = i; pt_random.y = j; pt_random.z = k;
            cloudMap.points.push_back(pt_random);
         }
      }
      // Side 2
      for (double j = trap_width/2; j < trap_width/2 + wall_thickness; j += _resolution) {
         for (double k = -1; k < _h_h; k += _resolution) {
            pt_random.x = i; pt_random.y = j; pt_random.z = k;
            cloudMap.points.push_back(pt_random);
         }
      }
   }

   // 3. Add a small obstacle inside the trap to create a Local Minima
   for (double i = trap_center_x + 2.0; i < trap_center_x + 3.0; i+= _resolution) {
       for (double j = -1.0; j < 1.0; j+= _resolution) {
           for (double k = -1; k < _h_h; k += _resolution) {
            pt_random.x = i; pt_random.y = j; pt_random.z = k;
            cloudMap.points.push_back(pt_random);
         }
       }
   }

   cloudMap.width = cloudMap.points.size();
   cloudMap.height = 1;
   cloudMap.is_dense = true;
   std::cout << "Generated Fixed Trap Map. Points: " << cloudMap.points.size() << std::endl;
   
   _has_map = true;
   pcl::toROSMsg(cloudMap, globalMap_pcd);
   globalMap_pcd.header.frame_id = "map";
}
void RandomMapGenerate()
{
   random_device rd;
   default_random_engine eng(rd());

   uniform_real_distribution<double> rand_theta = uniform_real_distribution<double>(-M_PI, M_PI);

   uniform_real_distribution<double> rand_x = uniform_real_distribution<double>(_x_l, _x_h);
   uniform_real_distribution<double> rand_y = uniform_real_distribution<double>(_y_l, _y_h);
   uniform_real_distribution<double> rand_w = uniform_real_distribution<double>(_w_l, _w_h);
   uniform_real_distribution<double> rand_h = uniform_real_distribution<double>(_h_l, _h_h);

   uniform_real_distribution<double> rand_x_circle = uniform_real_distribution<double>(_x_l + 1.0, _x_h - 1.0);
   uniform_real_distribution<double> rand_y_circle = uniform_real_distribution<double>(_y_l + 1.0, _y_h - 1.0);
   uniform_real_distribution<double> rand_r_circle = uniform_real_distribution<double>(_w_c_l, _w_c_h);

   uniform_real_distribution<double> rand_roll = uniform_real_distribution<double>(-M_PI, +M_PI);
   uniform_real_distribution<double> rand_pitch = uniform_real_distribution<double>(+M_PI / 4.0, +M_PI / 2.0);
   uniform_real_distribution<double> rand_yaw = uniform_real_distribution<double>(+M_PI / 4.0, +M_PI / 2.0);
   uniform_real_distribution<double> rand_ellipse_c = uniform_real_distribution<double>(0.5, 2.0);
   uniform_real_distribution<double> rand_num = uniform_real_distribution<double>(0.0, 1.0);

   pcl::PointXYZ pt_random;

   int base2(2), base3(3), base4(4); // Halton base
   // firstly, we put some circles
   for (int i = 0; i < _cir_num; i++)
   {
      double x0, y0, z0, R;
      std::vector<Vector3d> circle_set;

      // x0 = rand_x_circle(eng);
      // y0 = rand_y_circle(eng);
      z0 = rand_h(eng);

      // Halton sequence for x(0, 1)
      double f = 1;
      x0 = 0;
      int ii = i;
      while (ii > 0)
      {
         f = f / base2;
         x0 = x0 + f * (ii % base2);
         ii = floor(ii / base2);
      }
      x0 *= _x_size;
      x0 -= _x_size / 2;

      // Halton sequence for y(0, 1)
      f = 1;
      y0 = 0;
      ii = i;
      while (ii > 0)
      {
         f = f / base3;
         y0 = y0 + f * (ii % base3);
         ii = floor(ii / base3);
      }
      y0 *= _y_size;
      y0 -= _y_size / 2;

      R = rand_r_circle(eng);

      if (sqrt(pow(x0 - _init_x, 2) + pow(y0 - _init_y, 2)) < 1.5)
         continue;

      double a, b;
      a = rand_ellipse_c(eng);
      b = rand_ellipse_c(eng);

      double x, y, z;
      Vector3d pt3, pt3_rot;
      for (double theta = -M_PI; theta < M_PI; theta += 0.1)
      {
         for (double dr = -0.4; dr <= 0.4; dr += 0.4) {
            for (double dz = -0.4; dz <= 0.4; dz += 0.4) {
               x = a * cos(theta) * (R + dr);
               y = b * sin(theta) * (R + dr);
               z = dz;
               pt3 << x, y, z;
               circle_set.push_back(pt3);
            }
         }
      }
      // Define a random 3d rotation matrix
      Matrix3d Rot;
      double roll, pitch, yaw;
      double alpha, beta, gama;
      roll = rand_roll(eng);   // alpha
      pitch = rand_pitch(eng); // beta
      yaw = rand_yaw(eng);     // gama

      alpha = roll;
      beta = pitch;
      gama = yaw;

      double p = rand_num(eng);
      if (p < 0.5)
      {
         beta = M_PI / 2.0;
         gama = M_PI / 2.0;
      }

      Rot << cos(alpha) * cos(gama) - cos(beta) * sin(alpha) * sin(gama), -cos(beta) * cos(gama) * sin(alpha) - cos(alpha) * sin(gama), sin(alpha) * sin(beta),
          cos(gama) * sin(alpha) + cos(alpha) * cos(beta) * sin(gama), cos(alpha) * cos(beta) * cos(gama) - sin(alpha) * sin(gama), -cos(alpha) * sin(beta),
          sin(beta) * sin(gama), cos(gama) * sin(beta), cos(beta);

      for (auto pt : circle_set)
      {
         pt3_rot = Rot * pt;
         pt_random.x = pt3_rot(0) + x0 + 0.001;
         pt_random.y = pt3_rot(1) + y0 + 0.001;
         pt_random.z = pt3_rot(2) + z0 + 0.001 - 1;

         if (pt_random.z >= 0.0)
            cloudMap.points.push_back(pt_random);
      }
   }

   bool is_kdtree_empty = false;
   if (cloudMap.points.size() > 0)
      kdtreeMap.setInputCloud(cloudMap.makeShared());
   else
      is_kdtree_empty = true;

   // then, we put some pilar
   for (int i = 0; i < _obs_num; i++)
   {
      double x, y, w, h;
      // x    = rand_x(eng);
      // y    = rand_y(eng);
      w = rand_w(eng);

      // Halton sequence for x(0, 1)
      double f = 1;
      x = 0;
      int ii = i;
      while (ii > 0)
      {
         f = f / base2;
         x = x + f * (ii % base2);
         ii = floor(ii / base2);
      }
      x *= _x_size;
      x -= _x_size / 2;

      // Halton sequence for y(0, 1)
      f = 1;
      y = 0;
      ii = i;
      while (ii > 0)
      {
         f = f / base3;
         y = y + f * (ii % base3);
         ii = floor(ii / base3);
      }
      y *= _y_size;
      y -= _y_size / 2;

      double d_theta = rand_theta(eng);

      if (sqrt(pow(x - _init_x, 2) + pow(y - _init_y, 2)) < 2.0)
         continue;

      pcl::PointXYZ searchPoint(x, y, (_h_l + _h_h) / 2.0);
      pointIdxSearch.clear();
      pointSquaredDistance.clear();

      if (is_kdtree_empty == false)
      {
         if (kdtreeMap.nearestKSearch(searchPoint, 1, pointIdxSearch, pointSquaredDistance) > 0)
         {
            if (sqrt(pointSquaredDistance[0]) < 1.0)
               continue;
         }
      }

      x = floor(x / _resolution) * _resolution + _resolution / 2.0;
      y = floor(y / _resolution) * _resolution + _resolution / 2.0;

      int widNum = ceil(w / _resolution);
      int halfWidNum = widNum / 2.0;
      for (int r = -halfWidNum; r < halfWidNum; r++)
      {
         for (int s = -halfWidNum; s < halfWidNum; s++)
         {
            // make pilars hollow
            if (r > -halfWidNum + 2 && r < (halfWidNum - 3))
            {
               if (s > -halfWidNum + 2 && s < (halfWidNum - 3))
               {
                  continue;
               }
            }
            // rotate
            double th = atan2((double)s, (double)r);
            int len = sqrt(s * s + r * r);
            th += d_theta;
            int rr = cos(th) * len;
            int ss = sin(th) * len;

            h = rand_h(eng);
            int heiNum = 2.0 * ceil(h / _resolution);
            for (int t = 0; t < heiNum; t++)
            {
               pt_random.x = x + (rr + 0.0) * _resolution + 0.001;
               pt_random.y = y + (ss + 0.0) * _resolution + 0.001;
               pt_random.z = (t + 0.0) * _resolution * 0.5 - 1.0 + 0.001;
               cloudMap.points.push_back(pt_random);
            }
         }
      }
   }

   cloudMap.width = cloudMap.points.size();
   cloudMap.height = 1;
   cloudMap.is_dense = true;

   _has_map = true;

   pcl::toROSMsg(cloudMap, globalMap_pcd);
   globalMap_pcd.header.frame_id = "map";
}
// --- Environment V-Shape: Multiple Rectangular Obstacles in a V-formation (Adjustable Tip) ---
void GenerateEnvV_Shape()
{
    cloudMap.points.clear();
    pcl::PointXYZ pt;

    // Lambda function to add a rectangular block centered at (cx, cy)
    // Hàm lambda đã được sửa để nhận tâm (cx, cy) thay vì góc dưới bên trái, giúp dễ căn chỉnh hơn
    auto add_rect_block_centered = [&](double cx, double cy, double w, double h, double z_min, double z_max) {
        double x_start = cx - w / 2.0;
        double y_start = cy - h / 2.0;
        for (double i = x_start; i < x_start + w; i += _resolution)
            for (double j = y_start; j < y_start + h; j += _resolution)
                for (double k = z_min; k < z_max; k += _resolution) {
                    pt.x = i; pt.y = j; pt.z = k;
                    cloudMap.points.push_back(pt);
                }
    };

    // --- Parameters for the V-shape formation ---
    int num_obstacles_per_arm = 8;   // Tăng số lượng khối để nhìn dày hơn
    double block_width = 2.0;        // Chiều rộng ngang của mỗi khối
    double block_depth = 3.0;        // Chiều sâu dọc (theo trục Y) của mỗi khối
    double block_height = 2.5;       // Chiều cao

    double start_y = 12.0;           // Vị trí Y trên cùng (miệng chữ V)
    double end_y = -5.0;             // Vị trí Y dưới cùng (đáy nhọn chữ V)
    double start_x_span = 12.0;      // Khoảng cách X từ tâm ra đến miệng chữ V (độ mở phía trên)

    // --- NEW PARAMETER: Kiểm soát độ sát nhau ở đáy ---
    // tip_gap: Khoảng cách giữa tâm của 2 khối dưới cùng.
    // Nếu tip_gap = block_width (ví dụ 2.0), các cạnh của chúng sẽ vừa chạm nhau.
    // Nếu tip_gap < block_width (ví dụ 1.0), chúng sẽ chồng lấn lên nhau ở đáy, tạo cảm giác rất sát.
    double tip_gap = 3.0;
    double end_x_span = tip_gap / 2.0; // Khoảng cách X từ tâm ra đến khối đáy

    // Tính toán bước nhảy (step) để rải đều các khối
    // Sử dụng (num - 1) để đảm bảo khối đầu nằm đúng start_y và khối cuối nằm đúng end_y
    double y_step = (end_y - start_y) / (num_obstacles_per_arm - 1);

    // Left Arm Trajectory: từ (-start_x_span, start_y) đến (-end_x_span, end_y)
    double left_x_step = (-end_x_span - (-start_x_span)) / (num_obstacles_per_arm - 1);

    // Right Arm Trajectory: từ (start_x_span, start_y) đến (end_x_span, end_y)
    double right_x_step = (end_x_span - start_x_span) / (num_obstacles_per_arm - 1);


    // Generate Left Arm
    for (int i = 0; i < num_obstacles_per_arm; ++i) {
        double current_x = -start_x_span + i * left_x_step;
        double current_y = start_y + i * y_step;
        // Sử dụng hàm vẽ khối theo tâm đã sửa đổi
        add_rect_block_centered(current_x, current_y, block_width, block_depth, -1.0, block_height);
    }

    // Generate Right Arm
    for (int i = 0; i < num_obstacles_per_arm; ++i) {
        double current_x = start_x_span + i * right_x_step;
        double current_y = start_y + i * y_step;
        add_rect_block_centered(current_x, current_y, block_width, block_depth, -1.0, block_height);
    }

    cloudMap.width = cloudMap.points.size();
    cloudMap.height = 1;
    cloudMap.is_dense = true;
    _has_map = true;
    pcl::toROSMsg(cloudMap, globalMap_pcd);
    globalMap_pcd.header.frame_id = "map";
    ROS_INFO("Generated Tight V-Shape environment with %zu points.", cloudMap.points.size());
}
// --- Environment: 4 V-Shapes in a Line with Zig-Zag (Misaligned) Tips ---
void GenerateEnvMultiV_ZigZag()
{
    cloudMap.points.clear();
    pcl::PointXYZ pt;

    // 1. Hàm Lambda dùng chung để vẽ khối hộp
    auto add_rect_block_centered = [&](double cx, double cy, double w, double h, double z_min, double z_max) {
        double x_start = cx - w / 2.0;
        double y_start = cy - h / 2.0;
        for (double i = x_start; i < x_start + w; i += _resolution)
            for (double j = y_start; j < y_start + h; j += _resolution)
                for (double k = z_min; k < z_max; k += _resolution) {
                    pt.x = i; pt.y = j; pt.z = k;
                    cloudMap.points.push_back(pt);
                }
    };

    // 2. Tham số chung cho hình dáng mỗi chữ V
    int num_blocks_per_arm = 7;
    double block_w = 2.0;
    double block_d = 3.0;
    double block_h = 2.5;
    double v_longitudinal_len = 12.0; // Chiều dài dọc trục Y của mỗi chữ V
    double v_lateral_span = 10.0;     // Độ mở ngang (bán kính) ở miệng chữ V
    double tip_gap = 3.0;             // Khoảng cách giữa 2 khối ở đáy (khe hẹp)

    // 3. Cấu hình cho việc sắp xếp 4 chữ V
    int num_v_shapes = 4;
    double spacing_between_vs = 5.0; // Khoảng trống giữa đuôi chữ V trước và đầu chữ V sau
    double start_y_base = -30.0;     // Vị trí Y bắt đầu của hệ thống

    // === QUAN TRỌNG: Mảng chứa độ lệch (offset) của khe hẹp cho từng chữ V ===
    // Các giá trị này làm cho đỉnh chữ V lệch khỏi trục giữa (X=0).
    // Ví dụ: chữ V đầu tiên có đỉnh lệch sang phải 3m, chữ thứ 2 lệch sang trái 2.5m, v.v.
    std::vector<double> tip_offsets = {3.0, -2.5, 2.0, -1.5};

    double current_base_y = start_y_base;

    // 4. Vòng lặp để vẽ từng chữ V
    for (int n = 0; n < num_v_shapes; ++n) {
        // Tính toán vị trí Y đầu và cuối cho chữ V hiện tại
        double v_start_y = current_base_y + v_longitudinal_len; // Miệng chữ V (phía trên y)
        double v_end_y = current_base_y;                        // Đáy chữ V (phía dưới y)
        
        // Lấy độ lệch đỉnh cho chữ V này
        double current_tip_offset = tip_offsets[n % tip_offsets.size()];

        // Tính toán quỹ đạo cho 2 cánh tay
        // Tâm của hệ thống chữ V này bị dịch chuyển bởi current_tip_offset
        double start_x_left = current_tip_offset - v_lateral_span;
        double start_x_right = current_tip_offset + v_lateral_span;
        double end_x_left = current_tip_offset - tip_gap / 2.0;
        double end_x_right = current_tip_offset + tip_gap / 2.0;

        double y_step = (v_end_y - v_start_y) / (num_blocks_per_arm - 1);
        double left_x_step = (end_x_left - start_x_left) / (num_blocks_per_arm - 1);
        double right_x_step = (end_x_right - start_x_right) / (num_blocks_per_arm - 1);

        // Vẽ cánh tay trái
        for (int i = 0; i < num_blocks_per_arm; ++i) {
            add_rect_block_centered(start_x_left + i * left_x_step, v_start_y + i * y_step, block_w, block_d, -1.0, block_h);
        }
        // Vẽ cánh tay phải
        for (int i = 0; i < num_blocks_per_arm; ++i) {
            add_rect_block_centered(start_x_right + i * right_x_step, v_start_y + i * y_step, block_w, block_d, -1.0, block_h);
        }

        // Cập nhật vị trí Y cơ sở cho chữ V tiếp theo
        current_base_y += (v_longitudinal_len + spacing_between_vs);
    }

    cloudMap.width = cloudMap.points.size();
    cloudMap.height = 1;
    cloudMap.is_dense = true;
    _has_map = true;
    pcl::toROSMsg(cloudMap, globalMap_pcd);
    globalMap_pcd.header.frame_id = "map";
    ROS_INFO("Generated Multi-V Zig-Zag environment with %zu points.", cloudMap.points.size());
}
// --- Environment: Random Maze using DFS (Recursive Backtracking) ---
void GenerateEnv_Maze_DFS()
{
    cloudMap.points.clear();
    pcl::PointXYZ pt;
    ROS_INFO("Generating random maze map using DFS algorithm...");

    // 1. Cấu hình thông số Maze
    // Kích thước một ô của mê cung (độ rộng đường đi)
    double cell_size = 4.0;
    // Độ dày của tường
    double wall_thickness = 1.0;

    // Tính toán số lượng hàng và cột dựa trên kích thước bản đồ
    double map_width = _x_h - _x_l;
    double map_height = _y_h - _y_l;
    int cols = floor(map_width / cell_size);
    int rows = floor(map_height / cell_size);

    if (cols < 2 || rows < 2) {
        ROS_ERROR("Map size too small for maze generation!");
        return;
    }

    // Struct để lưu trạng thái từng ô
    struct MazeCell {
        bool visited = false;
        // Tường: Trên, Phải, Dưới, Trái. True là có tường.
        bool top = true; bool right = true; bool bottom = true; bool left = true;
    };

    // Tạo lưới mê cung
    std::vector<std::vector<MazeCell>> grid(rows, std::vector<MazeCell>(cols));

    // RNG setup
    std::random_device rd;
    std::mt19937 gen(rd());

    // 2. Thuật toán DFS (Iterative using Stack) để tạo Maze
    std::stack<std::pair<int, int>> stack;
    // Bắt đầu từ ô (0,0)
    int current_r = 0;
    int current_c = 0;
    grid[current_r][current_c].visited = true;
    stack.push({current_r, current_c});

    while (!stack.empty()) {
        std::pair<int, int> current = stack.top();
        int r = current.first;
        int c = current.second;

        // Tìm hàng xóm chưa thăm
        std::vector<int> neighbors; // 0:Top, 1:Right, 2:Bottom, 3:Left
        if (r > 0 && !grid[r - 1][c].visited) neighbors.push_back(0);
        if (c < cols - 1 && !grid[r][c + 1].visited) neighbors.push_back(1);
        if (r < rows - 1 && !grid[r + 1][c].visited) neighbors.push_back(2);
        if (c > 0 && !grid[r][c - 1].visited) neighbors.push_back(3);

        if (!neighbors.empty()) {
            // Chọn ngẫu nhiên một hàng xóm
            std::uniform_int_distribution<> dist(0, neighbors.size() - 1);
            int next_dir = neighbors[dist(gen)];
            int next_r = r, next_c = c;

            // Phá tường giữa ô hiện tại và hàng xóm
            if (next_dir == 0) { // Top
                grid[r][c].top = false; grid[r - 1][c].bottom = false;
                next_r--;
            } else if (next_dir == 1) { // Right
                grid[r][c].right = false; grid[r][c + 1].left = false;
                next_c++;
            } else if (next_dir == 2) { // Bottom
                grid[r][c].bottom = false; grid[r + 1][c].top = false;
                next_r++;
            } else if (next_dir == 3) { // Left
                grid[r][c].left = false; grid[r][c - 1].right = false;
                next_c--;
            }

            grid[next_r][next_c].visited = true;
            stack.push({next_r, next_c});
        } else {
            // Không còn hàng xóm, quay lui (backtrack)
            stack.pop();
        }
    }

    // 3. Mở lối vào (Start) và lối ra (Goal)
    // Ví dụ: Vào ở góc dưới trái, ra ở góc trên phải
    grid[0][0].left = false; // Mở tường trái ô đầu tiên
    grid[rows - 1][cols - 1].right = false; // Mở tường phải ô cuối cùng


    // 4. Chuyển đổi Grid thành PointCloud (Vẽ tường)
    // Helper lambda để vẽ một khối chữ nhật
    auto add_wall_block = [&](double cx, double cy, double w, double h) {
        double x_start = cx - w / 2.0;
        double y_start = cy - h / 2.0;
        for (double i = x_start; i < x_start + w; i += _resolution)
            for (double j = y_start; j < y_start + h; j += _resolution)
                for (double k = -1.0; k < _h_h; k += _resolution) {
                    pt.x = i; pt.y = j; pt.z = k;
                    cloudMap.points.push_back(pt);
                }
    };

    // Tính toán gốc tọa độ để vẽ (góc dưới trái của lưới)
    double start_x_phys = _x_l + (map_width - cols * cell_size) / 2.0;
    double start_y_phys = _y_l + (map_height - rows * cell_size) / 2.0;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            // Tâm vật lý của ô hiện tại
            double cell_cx = start_x_phys + c * cell_size + cell_size / 2.0;
            double cell_cy = start_y_phys + r * cell_size + cell_size / 2.0;

            // Để tránh vẽ trùng lặp, ta chỉ vẽ tường DƯỚI và tường PHẢI của mỗi ô.
            // Tường TRÊN và tường TRÁI của biên bản đồ sẽ được vẽ riêng.

            if (grid[r][c].bottom) {
                // Vẽ tường dưới: tâm nằm ở cạnh dưới ô, rộng=cell_size, cao=thickness
                add_wall_block(cell_cx, cell_cy - cell_size / 2.0, cell_size + wall_thickness, wall_thickness);
            }
            if (grid[r][c].right) {
                // Vẽ tường phải: tâm nằm ở cạnh phải ô, rộng=thickness, cao=cell_size
                add_wall_block(cell_cx + cell_size / 2.0, cell_cy, wall_thickness, cell_size + wall_thickness);
            }

            // Vẽ biên trái cùng
            if (c == 0 && grid[r][c].left) {
                 add_wall_block(cell_cx - cell_size / 2.0, cell_cy, wall_thickness, cell_size + wall_thickness);
            }
            // Vẽ biên trên cùng
            if (r == rows - 1 && grid[r][c].top) {
                 add_wall_block(cell_cx, cell_cy + cell_size / 2.0, cell_size + wall_thickness, wall_thickness);
            }
        }
    }

    cloudMap.width = cloudMap.points.size();
    cloudMap.height = 1;
    cloudMap.is_dense = true;
    _has_map = true;
    pcl::toROSMsg(cloudMap, globalMap_pcd);
    globalMap_pcd.header.frame_id = "map";
    ROS_INFO("Generated Maze map with %zu points. Grid size: %dx%d", cloudMap.points.size(), rows, cols);
}
// Add this function to your map generation code
void GenerateClutterTrap()
{
   cloudMap.points.clear();
   pcl::PointXYZ pt;
   
   // 1. Clear Start and Goal areas
   // Assuming Start is near (-20, 0) and Goal is near (20, 0)

   // 2. Create a "Broken Wall" at X = 0
   // This forces the robot to weave through or go around, triggering heuristic issues.
   double wall_x = 0.0;
   double block_size = 2.0;
   double gap = 0.5; // Small gap that might be too narrow, creating local minima
   
   // Create a vertical line of blocks with small gaps
   for (double y = -15.0; y <= 15.0; y += (block_size + gap)) {
       
       // Randomly shift x slightly to create "clutter" feel
       double current_x = wall_x + ((rand() % 10) / 10.0); 

       for (double i = current_x; i < current_x + block_size; i += _resolution)
           for (double j = y; j < y + block_size; j += _resolution)
               for (double k = -1; k < _h_h; k += _resolution) {
                   pt.x = i; pt.y = j; pt.z = k;
                   cloudMap.points.push_back(pt);
               }
   }

   cloudMap.width = cloudMap.points.size();
   cloudMap.height = 1;
   cloudMap.is_dense = true;
   _has_map = true;
   pcl::toROSMsg(cloudMap, globalMap_pcd);
   globalMap_pcd.header.frame_id = "map";
}
// -----------------------------------------------------------------------
// Type3: Narrow
// Matches image: large rectangular blocks on top and bottom halves with
// a narrow horizontal corridor between them, offset each column to force
// slalom navigation.  3 columns of block-pairs across the map.
// Path: enter bottom-left gap → cross top-right gap → exit.
// -----------------------------------------------------------------------
// Type3: Narrow passage — matches reference image exactly.
// 4 full-height vertical black bars create 5 narrow passages (left edge,
// 3 between bars, right edge).  One small square obstacle sits in each
// passage at a descending height, producing a top→bottom slalom.
// A clear path always exists beside each obstacle.
// -----------------------------------------------------------------------
void GenerateType3_Narrow()
{
    cloudMap.points.clear();
    pcl::PointXYZ pt;

    double s   = (_x_h - _x_l) / 100.0;
    double res = _resolution;

    // Fill a full-height vertical column between x0 and x1
    auto fill_col = [&](double x0, double x1) {
        for (double x = x0; x < x1; x += res)
            for (double y = _y_l; y < _y_h; y += res)
                for (double z = -1.0; z < _h_h; z += res) {
                    pt.x = x; pt.y = y; pt.z = z;
                    cloudMap.points.push_back(pt);
                }
    };

    // Fill a sq×sq obstacle centred at (cx, cy)
    auto small_obs = [&](double cx, double cy, double sq) {
        for (double x = cx - sq/2.0; x < cx + sq/2.0; x += res)
            for (double y = cy - sq/2.0; y < cy + sq/2.0; y += res)
                for (double z = -1.0; z < _h_h; z += res) {
                    pt.x = x; pt.y = y; pt.z = z;
                    cloudMap.points.push_back(pt);
                }
    };

    // ── Layout (all in normalised ±50 coords, scaled by s) ───────────────
    // left-edge passage | Bar1 | passage1 | Bar2 | passage2 | Bar3 | passage3 | Bar4 | right-edge passage
    //       4 u              18 u    5 u      18 u    5 u      18 u    5 u      18 u       9 u   = 100 u
    double bar_w  = 18.0 * s;
    double pas_w  =  5.0 * s;
    double ledge  =  4.0 * s;   // left-edge passage width
    double redge  =  9.0 * s;   // right-edge passage width (wider to sum to 100)

    // Bar x-starts
    double b1x = _x_l + ledge;
    double b2x = b1x + bar_w + pas_w;
    double b3x = b2x + bar_w + pas_w;
    double b4x = b3x + bar_w + pas_w;

    fill_col(b1x, b1x + bar_w);
    fill_col(b2x, b2x + bar_w);
    fill_col(b3x, b3x + bar_w);
    fill_col(b4x, b4x + bar_w);

    // ── Passage centre x positions ────────────────────────────────────────
    double px0 = _x_l + ledge / 2.0;               // left-edge passage
    double px1 = b1x + bar_w + pas_w / 2.0;        // passage 1
    double px2 = b2x + bar_w + pas_w / 2.0;        // passage 2
    double px3 = b3x + bar_w + pas_w / 2.0;        // passage 3
    double px4 = b4x + bar_w + redge / 2.0;        // right-edge passage

    // ── Small obstacles — one per passage, descending y (top→bottom slalom) ──
    double sq = 3.0 * s;   // obstacle side (much smaller than passage width)
    small_obs(px0,  35.0 * s, sq);   // left edge  — near top
    small_obs(px1,  17.0 * s, sq);   // passage 1  — upper-middle
    small_obs(px2,  -3.0 * s, sq);   // passage 2  — centre
    small_obs(px3, -22.0 * s, sq);   // passage 3  — lower-middle
    small_obs(px4, -38.0 * s, sq);   // right edge — near bottom

    cloudMap.width = cloudMap.points.size();
    cloudMap.height = 1; cloudMap.is_dense = true; _has_map = true;
    pcl::toROSMsg(cloudMap, globalMap_pcd);
    globalMap_pcd.header.frame_id = "map";
    ROS_INFO("Generated Type3 Narrow map with %zu points.", cloudMap.points.size());
}

// -----------------------------------------------------------------------
// Type4: Rooms
// Matches image: a building floor-plan with rooms of varying sizes.
// Fixed 3×2 layout (3 columns, 2 rows).  Every pair of adjacent rooms
// has one doorway at a fixed, offset position so all rooms are reachable.
// Outer boundary walls are included.
// -----------------------------------------------------------------------
void GenerateType4_Rooms()
{
    cloudMap.points.clear();
    pcl::PointXYZ pt;

    double s   = (_x_h - _x_l) / 100.0;
    double t   = 2.0 * s;    // wall thickness
    double d   = 7.0 * s;    // doorway width
    double res = _resolution;

    // Helper: fill a solid rectangle (corner at x0,y0, size wx × wy)
    auto fill_rect = [&](double x0, double y0, double wx, double wy) {
        for (double x = x0; x < x0 + wx; x += res)
            for (double y = y0; y < y0 + wy; y += res)
                for (double z = -1.0; z < _h_h; z += res) {
                    pt.x = x; pt.y = y; pt.z = z;
                    cloudMap.points.push_back(pt);
                }
    };

    // Helper: horizontal wall from (x0,y) to (x1,y) with door centred at door_cx
    auto h_wall = [&](double x0, double x1, double y, double door_cx) {
        double d0 = door_cx - d / 2.0, d1 = door_cx + d / 2.0;
        for (double x = x0; x < x1; x += res) {
            if (x >= d0 && x <= d1) continue;
            for (double yy = y; yy < y + t; yy += res)
                for (double z = -1.0; z < _h_h; z += res) {
                    pt.x = x; pt.y = yy; pt.z = z;
                    cloudMap.points.push_back(pt);
                }
        }
    };

    // Helper: vertical wall from (x,y0) to (x,y1) with door centred at door_cy
    auto v_wall = [&](double x, double y0, double y1, double door_cy) {
        double d0 = door_cy - d / 2.0, d1 = door_cy + d / 2.0;
        for (double y = y0; y < y1; y += res) {
            if (y >= d0 && y <= d1) continue;
            for (double xx = x; xx < x + t; xx += res)
                for (double z = -1.0; z < _h_h; z += res) {
                    pt.x = xx; pt.y = y; pt.z = z;
                    cloudMap.points.push_back(pt);
                }
        }
    };

    // Map bounds
    double X0 = _x_l, X3 = _x_h;
    double Y0 = _y_l, Y2 = _y_h;

    // Column dividers (x positions): 3 columns → 2 vertical dividers
    double X1 = X0 + (_x_h - _x_l) * 0.38;   // first divider (rooms vary in width)
    double X2 = X0 + (_x_h - _x_l) * 0.65;   // second divider

    // Row divider (y position): 2 rows → 1 horizontal divider
    double Y1 = Y0 + (_y_h - _y_l) * 0.45;   // slightly below centre

    // Outer boundary (solid walls, no doors on boundary)
    fill_rect(X0, Y0, X3 - X0, t);             // bottom
    fill_rect(X0, Y2 - t, X3 - X0, t);         // top
    fill_rect(X0, Y0, t, Y2 - Y0);             // left
    fill_rect(X3 - t, Y0, t, Y2 - Y0);         // right

    // Horizontal divider (row 0 → row 1) — door positions offset per column
    double hd0 = X0 + (X1 - X0) * 0.30;   // col-0 door centre x
    double hd1 = X1 + (X2 - X1) * 0.65;   // col-1 door centre x
    double hd2 = X2 + (X3 - X2) * 0.40;   // col-2 door centre x
    h_wall(X0, X1, Y1, hd0);
    h_wall(X1, X2, Y1, hd1);
    h_wall(X2, X3, Y1, hd2);

    // Vertical divider 1 (col 0 → col 1) — door per row
    double vd1_bot = Y0 + (Y1 - Y0) * 0.55;   // bottom row door centre y
    double vd1_top = Y1 + (Y2 - Y1) * 0.35;   // top row door centre y
    v_wall(X1, Y0, Y1, vd1_bot);
    v_wall(X1, Y1, Y2, vd1_top);

    // Vertical divider 2 (col 1 → col 2) — door per row
    double vd2_bot = Y0 + (Y1 - Y0) * 0.70;   // bottom row door centre y
    double vd2_top = Y1 + (Y2 - Y1) * 0.50;   // top row door centre y
    v_wall(X2, Y0, Y1, vd2_bot);
    v_wall(X2, Y1, Y2, vd2_top);

    // Square baffles in front of every doorway — on both room faces of each wall.
    // sq < d (door width) so a gap of (d-sq)/2 = 1.75s remains on each side.
    double sq  = 3.5 * s;   // baffle side length
    double gap = 1.5 * s;   // clear space between wall face and nearest baffle edge

    // centre helper: places a sq×sq obstacle centred at (cx, cy)
    auto baffle = [&](double cx, double cy) {
        fill_rect(cx - sq / 2.0, cy - sq / 2.0, sq, sq);
    };

    // H-wall baffles — lower room (y < Y1) and upper room (y > Y1+t)
    double by_lo = Y1 - gap - sq / 2.0;         // centre y in lower room
    double by_hi = Y1 + t + gap + sq / 2.0;     // centre y in upper room
    baffle(hd0, by_lo);  baffle(hd0, by_hi);    // col-0 door
    baffle(hd1, by_lo);  baffle(hd1, by_hi);    // col-1 door
    baffle(hd2, by_lo);  baffle(hd2, by_hi);    // col-2 door

    // V-wall 1 baffles — left room (x < X1) and right room (x > X1+t)
    double bx1_lo = X1 - gap - sq / 2.0;        // centre x in left room
    double bx1_hi = X1 + t + gap + sq / 2.0;    // centre x in right room
    baffle(bx1_lo, vd1_bot);  baffle(bx1_hi, vd1_bot);   // bottom row door
    baffle(bx1_lo, vd1_top);  baffle(bx1_hi, vd1_top);   // top row door

    // V-wall 2 baffles — left room (x < X2) and right room (x > X2+t)
    double bx2_lo = X2 - gap - sq / 2.0;
    double bx2_hi = X2 + t + gap + sq / 2.0;
    baffle(bx2_lo, vd2_bot);  baffle(bx2_hi, vd2_bot);   // bottom row door
    baffle(bx2_lo, vd2_top);  baffle(bx2_hi, vd2_top);   // top row door

    cloudMap.width = cloudMap.points.size();
    cloudMap.height = 1; cloudMap.is_dense = true; _has_map = true;
    pcl::toROSMsg(cloudMap, globalMap_pcd);
    globalMap_pcd.header.frame_id = "map";
    ROS_INFO("Generated Type4 Rooms map with %zu points.", cloudMap.points.size());
}

// -----------------------------------------------------------------------
// Type5: Blobs
// Matches image: 9 large filled circular blobs spread across the map in a
// 3×3 grid pattern, with each blob offset from its grid centre to break
// regularity.  Corridors between blobs are always ≥ 8 units wide.
// -----------------------------------------------------------------------
void GenerateType5_Blobs()
{
    cloudMap.points.clear();
    pcl::PointXYZ pt;

    double s   = (_x_h - _x_l) / 100.0;
    double res = _resolution;

    // 9 blobs in a 3×3 arrangement; offsets break grid regularity.
    // Positions in normalised units (map spans −50…+50 before scaling).
    struct Blob { double cx, cy, r; };
    std::vector<Blob> blobs = {
        // bottom row
        { -33, -35,  9 },
        {   2, -38,  8 },
        {  36, -33,  9 },
        // middle row
        { -38,   0,  8 },
        {  -5,   5,  7 },
        {  35,   2,  8 },
        // top row
        { -32,  34,  9 },
        {   5,  37,  8 },
        {  34,  33,  9 },
    };

    for (auto& b : blobs) {
        double cx = b.cx * s, cy = b.cy * s, R = b.r * s;
        for (double x = cx - R; x <= cx + R; x += res)
            for (double y = cy - R; y <= cy + R; y += res) {
                if ((x-cx)*(x-cx) + (y-cy)*(y-cy) > R*R) continue;
                for (double z = -1.0; z < _h_h; z += res) {
                    pt.x = x; pt.y = y; pt.z = z;
                    cloudMap.points.push_back(pt);
                }
            }
    }

    cloudMap.width = cloudMap.points.size();
    cloudMap.height = 1; cloudMap.is_dense = true; _has_map = true;
    pcl::toROSMsg(cloudMap, globalMap_pcd);
    globalMap_pcd.header.frame_id = "map";
    ROS_INFO("Generated Type5 Blobs map with %zu points.", cloudMap.points.size());
}

// -----------------------------------------------------------------------
// Type6: Flappy style
// 4 full-height vertical barriers (like Flappy Bird pipes) spanning the
// entire map height, each with one opening flush to the map boundary.
// Openings alternate strictly bottom / top / bottom / top so a single
// zigzag path threads through all 4 barriers.
// -----------------------------------------------------------------------
void GenerateType6_Flappy()
{
    cloudMap.points.clear();
    pcl::PointXYZ pt;

    double s      = (_x_h - _x_l) / 100.0;
    double res    = _resolution;
    double pipe_w = 12.0 * s;   // barrier width ≈ 12% of map
    double gap_h  = 35.0 * s;   // opening height ≈ 35% of map (flush to edge)

    // 4 evenly-spaced barriers
    int    n_pipes = 4;
    double spacing = (_x_h - _x_l) / (n_pipes + 1);

    for (int p = 0; p < n_pipes; ++p) {
        double pipe_x = _x_l + spacing * (p + 1) - pipe_w / 2.0;

        // Opening flush to bottom edge (even pipes) or top edge (odd pipes)
        double gap_lo, gap_hi;
        if (p % 2 == 0) {
            gap_lo = _y_l;
            gap_hi = _y_l + gap_h;
        } else {
            gap_lo = _y_h - gap_h;
            gap_hi = _y_h;
        }

        for (double x = pipe_x; x < pipe_x + pipe_w; x += res)
            for (double y = _y_l; y < _y_h; y += res) {
                if (y >= gap_lo && y < gap_hi) continue;
                for (double z = -1.0; z < _h_h; z += res) {
                    pt.x = x; pt.y = y; pt.z = z;
                    cloudMap.points.push_back(pt);
                }
            }
    }

    // Small square obstacles scattered in the gap openings and entry/exit
    // corridors.  Coordinates are in normalised units (map spans -50..+50),
    // scaled by s.  hw = half-width of each square pillar.
    struct SmObs { double cx, cy, hw; };
    const std::vector<SmObs> extras = {
        // entry corridor (before pipe 0, x ≈ -44)
        { -43,  22, 2.0 }, { -44, -30, 2.0 },
        // pipe 0 bottom opening (x ≈ -30, y: -50 to -15)
        { -30, -42, 2.0 }, { -30, -23, 2.0 },
        // pipe 1 top opening (x ≈ -10, y: +15 to +50)
        { -10,  27, 2.0 }, { -10,  43, 2.0 },
        // pipe 2 bottom opening (x ≈ +10, y: -50 to -15)
        {  10, -38, 2.0 }, {  10, -21, 2.0 },
        // pipe 3 top opening (x ≈ +30, y: +15 to +50)
        {  30,  24, 2.0 }, {  30,  43, 2.0 },
        // exit corridor (after pipe 3, x ≈ +44)
        {  43, -22, 2.0 }, {  44,  30, 2.0 },
    };
    for (const auto& o : extras) {
        double cx = o.cx * s, cy = o.cy * s, hw = o.hw * s;
        for (double x = cx - hw; x < cx + hw; x += res)
            for (double y = cy - hw; y < cy + hw; y += res)
                for (double z = -1.0; z < _h_h; z += res) {
                    pt.x = x; pt.y = y; pt.z = z;
                    cloudMap.points.push_back(pt);
                }
    }

    cloudMap.width = cloudMap.points.size();
    cloudMap.height = 1; cloudMap.is_dense = true; _has_map = true;
    pcl::toROSMsg(cloudMap, globalMap_pcd);
    globalMap_pcd.header.frame_id = "map";
    ROS_INFO("Generated Type6 Flappy map with %zu points.", cloudMap.points.size());
}

// -----------------------------------------------------------------------
// Type7: U-shaped obstacles
// Matches image: mix of small, medium, and large U-shapes at fixed
// positions and orientations scattered across the map.  The U openings
// face various directions so a planner must navigate around or through
// them.  Corridors between shapes are always ≥ 6 units wide.
// -----------------------------------------------------------------------
void GenerateType7_UShaped()
{
    cloudMap.points.clear();
    pcl::PointXYZ pt;

    double s   = (_x_h - _x_l) / 100.0;
    double res = _resolution;

    auto fill_rect = [&](double cx, double cy, double w, double h) {
        for (double x = cx - w/2.0; x < cx + w/2.0; x += res)
            for (double y = cy - h/2.0; y < cy + h/2.0; y += res)
                for (double z = -1.0; z < _h_h; z += res) {
                    pt.x = x; pt.y = y; pt.z = z;
                    cloudMap.points.push_back(pt);
                }
    };

    // Draw one U-shape: cx,cy = centre, L = arm length, W = opening width,
    // t = wall thickness, opening: 0=right 1=left 2=up 3=down
    auto draw_u = [&](double cx, double cy, double L, double W, double t, int opening) {
        switch (opening) {
        case 0: // open right: base on left, two arms extend right
            fill_rect(cx - L/2.0, cy,            t, W    );   // base (left)
            fill_rect(cx,          cy + W/2.0 - t/2.0, L, t);   // top arm
            fill_rect(cx,          cy - W/2.0 + t/2.0, L, t);   // bottom arm
            break;
        case 1: // open left: base on right
            fill_rect(cx + L/2.0, cy,            t, W    );
            fill_rect(cx,          cy + W/2.0 - t/2.0, L, t);
            fill_rect(cx,          cy - W/2.0 + t/2.0, L, t);
            break;
        case 2: // open up: base on bottom
            fill_rect(cx, cy - L/2.0,            W, t    );
            fill_rect(cx + W/2.0 - t/2.0, cy,    t, L    );
            fill_rect(cx - W/2.0 + t/2.0, cy,    t, L    );
            break;
        default: // open down: base on top
            fill_rect(cx, cy + L/2.0,            W, t    );
            fill_rect(cx + W/2.0 - t/2.0, cy,    t, L    );
            fill_rect(cx - W/2.0 + t/2.0, cy,    t, L    );
            break;
        }
    };

    // Fixed layout — (cx, cy) in normalised ±50 coords, then scaled.
    // Columns: small (S), medium (M), large (L) U-shapes mixed in.
    // opening: 0=right 1=left 2=up 3=down
    struct UDef { double cx, cy, arm_len, width, thick; int opening; };
    std::vector<UDef> defs = {
        // top-left cluster
        { -38,  38,  12, 10, 2.0, 0 },   // large, open right
        { -18,  40,   7,  6, 1.5, 3 },   // small, open down
        { -38,  18,   8,  7, 1.5, 2 },   // medium, open up
        { -20,  22,  10,  8, 2.0, 1 },   // medium, open left

        // top-right cluster
        {  38,  38,  12, 10, 2.0, 1 },   // large, open left
        {  18,  40,   7,  6, 1.5, 3 },   // small, open down
        {  38,  18,   8,  7, 1.5, 2 },   // medium, open up
        {  20,  22,  10,  8, 2.0, 0 },   // medium, open right

        // bottom-left cluster
        { -38, -38,  12, 10, 2.0, 0 },   // large, open right
        { -18, -40,   7,  6, 1.5, 2 },   // small, open up
        { -38, -18,   8,  7, 1.5, 3 },   // medium, open down
        { -20, -22,  10,  8, 2.0, 1 },   // medium, open left

        // bottom-right cluster
        {  38, -38,  12, 10, 2.0, 1 },   // large, open left
        {  18, -40,   7,  6, 1.5, 2 },   // small, open up
        {  38, -18,   8,  7, 1.5, 3 },   // medium, open down
        {  20, -22,  10,  8, 2.0, 0 },   // medium, open right
    };

    for (auto& d : defs)
        draw_u(d.cx * s, d.cy * s, d.arm_len * s, d.width * s, d.thick * s, d.opening);

    cloudMap.width = cloudMap.points.size();
    cloudMap.height = 1; cloudMap.is_dense = true; _has_map = true;
    pcl::toROSMsg(cloudMap, globalMap_pcd);
    globalMap_pcd.header.frame_id = "map";
    ROS_INFO("Generated Type7 U-Shaped map with %zu points.", cloudMap.points.size());
}

void pubSensedPoints()
{
   if (!_has_map)
      return;
   _all_map_pub.publish(globalMap_pcd);
}

bool pubGlbObs(self_msgs_and_srvs::GlbObsRcv::Request &req, self_msgs_and_srvs::GlbObsRcv::Response &res)
{
   pubSensedPoints();
   return true;
}

int main(int argc, char **argv)
{
   ros::init(argc, argv, "random_map_sensing");
   ros::NodeHandle n("~");

   _all_map_pub = n.advertise<sensor_msgs::PointCloud2>("all_map", 1);

   n.param("init_state_x", _init_x, 0.0);
   n.param("init_state_y", _init_y, 0.0);

   n.param("map/x_size", _x_size, 50.0);
   n.param("map/y_size", _y_size, 50.0);
   n.param("map/z_size", _z_size, 5.0);

   n.param("map/obs_num", _obs_num, 30);
   n.param("map/circle_num", _cir_num, 30);
   n.param("map/resolution", _resolution, 0.2);

   n.param("ObstacleShape/lower_rad", _w_l, 0.3);
   n.param("ObstacleShape/upper_rad", _w_h, 0.8);
   n.param("ObstacleShape/lower_hei", _h_l, 3.0);
   n.param("ObstacleShape/upper_hei", _h_h, 7.0);

   n.param("CircleShape/lower_circle_rad", _w_c_l, 0.3);
   n.param("CircleShape/upper_circle_rad", _w_c_h, 0.8);

   n.param("sensing/rate", _sense_rate, 1.0);

   _x_l = -_x_size / 2.0;
   _x_h = +_x_size / 2.0;

   _y_l = -_y_size / 2.0;
   _y_h = +_y_size / 2.0;
      // === Map Generation Selection ===
   std::string map_type;
   n.param("map/map_type", map_type, std::string("random_large"));

   ROS_INFO("Selected map type: %s", map_type.c_str());

   if (map_type == "fixed")
   {
      FixedTrapAndNarrowGenerate();
   }
   else if (map_type == "env_a") {
      GenerateEnvA_Simple();
   }
   else if (map_type == "env_b") {
      GenerateEnvB_Cluttered();
   }
   else if (map_type == "env_c") {
      GenerateEnvC_Maze();
   }
   else if (map_type == "narrow_map") {
      GenerateEnvD_Narrow();
   }
   else if (map_type == "l_shape_grid") {
      GenerateEnv_L_Shape_Grid();
   }
   else if (map_type == "maze_dfs") {
      GenerateEnv_Maze_DFS();
   }
   else if (map_type == "random_large")
   {
      RandomBRRTGenerate_Large();
   }
   else if (map_type == "random_buildings")
   {
      RandomBRRTGenerate_Buildings();
   }
   else if (map_type == "random_cubes")
   {
      RandomBRRTGenerate_Cubes();
   }
   else if (map_type == "robot_arm")
   {
      RobotArmMapGenerate();
   }
   else if (map_type == "multipe"){
      FixedMapGenerate();
   }
   else if (map_type == "random_narrow")
   {
      RandomNarrowGenerate();
   }
   else if (map_type == "random")
   {
      RandomMapGenerate();
   }
   else if (map_type == "zju_sample")
   {
      ROS_INFO("Generating ZJU-FAST-Lab sample map with specific parameters scaled to map size.");
      // Maintain the ZJU map density (120 obs and 100 circles per 50x50 area)
      double area = _x_size * _y_size;
      _obs_num = static_cast<int>(area * 0.048); // reasonable density
      _cir_num = static_cast<int>(area * 0.040); // reasonable density
      
      _w_l = 2.0; _w_h = 10.0;
      _h_l = 3.0; _h_h = 15.0;
      _w_c_l = 4.0; _w_c_h = 10.0;
      RandomMapGenerate();
   }
   else if (map_type == "random_brrt")
   {
      RandomBRRTGenerate();
   }
   else if (map_type == "env_v_shape") {
      GenerateEnvV_Shape();
   }
   else if (map_type == "env_multi_v_zigzag") {
      GenerateEnvMultiV_ZigZag();
   }
   else if (map_type == "clutter_trap") {
      GenerateClutterTrap();
   }
   else if (map_type == "narrow") {
      GenerateType3_Narrow();
   }
   else if (map_type == "rooms") {
      GenerateType4_Rooms();
   }
   else if (map_type == "blobs") {
      GenerateType5_Blobs();
   }
   else if (map_type == "flappy") {
      GenerateType6_Flappy();
   }
   else if (map_type == "u_shaped") {
      GenerateType7_UShaped();
   }
   else
   {
      ROS_ERROR("Unknown map type: '%s'. Defaulting to 'random_large'.", map_type.c_str());
      RandomBRRTGenerate_Large();
   }
   // =================================
   // RandomMapGenerate();
   // RandomNarrowGenerate();
   // RandomBRRTGenerate_Large();
   // only pub map pointcloud on request
   ros::ServiceServer pub_glb_obs_service = n.advertiseService("/pub_glb_obs", pubGlbObs);
   ros::spin();

   // ros::Rate loop_rate(_sense_rate);
   // while (ros::ok())
   // {
   //    ros::spinOnce();
   //    loop_rate.sleep();
   // }
}
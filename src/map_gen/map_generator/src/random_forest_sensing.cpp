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

void RandomBRRTGenerate_Large(double size = 4)
{
   pcl::PointXYZ pt_random;
   random_device rd;
   default_random_engine eng(rd());
   float ramdom_ratio = 0.6;
   int number_ostacle = (_x_h - _x_l) * (_y_h - _y_l) / (size * size) * ramdom_ratio;
   std::cout << "number of ostacle" << number_ostacle;

   std::mt19937 gen(rd()); // seed the generator

   // Create distribution in range [a, b]
   std::uniform_real_distribution<> dis_x(_x_l, _x_h);
   std::uniform_real_distribution<> dis_y(_y_l, _y_h);
   // Generate a random number
   double half_size = size / 2;
   for (int i = 0; i < number_ostacle; i++)
   {
      double random_x = dis_x(gen);
      double random_y = dis_y(gen);
      for (double i_x = random_x - half_size; i_x < random_x + half_size; i_x += 0.5)
         for (double i_y = random_y - half_size; i_y < random_y + half_size; i_y += 0.5)
         for (float k = -1; k < _h_h; k+=0.5)
         {
            pt_random.x = i_x;
            pt_random.y = i_y;
            pt_random.z = k;
            cloudMap.points.push_back(pt_random);
         }
   }

   // pcl::PointXYZ pt_random;
   // std::cout<<"size of map" << _x_l << " " << _x_h << " " << _y_l << " " << _y_h <<" " << _h_h <<std::endl;
   // // generate  1000 points random with size 4
   // for (float i = _x_l; i < _x_h; i += size)
   // {
   //    for (float j = _y_l; j < _y_h; j += size)
   //    {
   //       // get a random number between 0 and 1
   //       float random_num = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
   //       if (random_num < ramdom_ratio)
   //       {
   //          for (float k = -1; k < _h_h; k += size)
   //          {
   //             pt_random.x = i;
   //             pt_random.y = j;
   //             pt_random.z = k;
   //             cloudMap.points.push_back(pt_random);
   //          }
   //       }
   //    }
   // }

   cloudMap.width = cloudMap.points.size();
   cloudMap.height = 1;
   cloudMap.is_dense = true;
   std::cout << "cloudMap.points.size() = " << cloudMap.points.size() << std::endl;
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

   double wall_x = 0.0;     // Position of the wall
   double wall_width = 3.0; // Thickness of the wall
   double gap_size = 1.5;   // Size of the narrow passage

   // Lower Wall
   for (double i = wall_x - wall_width/2; i < wall_x + wall_width/2; i += _resolution)
      for (double j = _y_l; j < -gap_size/2; j += _resolution)
         for (double k = -1; k < _h_h; k += _resolution) {
            pt.x = i; pt.y = j; pt.z = k;
            cloudMap.points.push_back(pt);
         }

   // Upper Wall
   for (double i = wall_x - wall_width/2; i < wall_x + wall_width/2; i += _resolution)
      for (double j = gap_size/2; j < _y_h; j += _resolution)
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
      for (double theta = -M_PI; theta < M_PI; theta += 0.025)
      {
         x = a * cos(theta) * R;
         y = b * sin(theta) * R;
         z = 0;
         pt3 << x, y, z;
         circle_set.push_back(pt3);
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
   else if (map_type == "env_d") {
      GenerateEnvD_Narrow();
   }
   else if (map_type == "random_large")
   {
      RandomBRRTGenerate_Large();
   }
   else if (map_type == "random_narrow")
   {
      RandomNarrowGenerate();
   }
   else if (map_type == "random")
   {
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
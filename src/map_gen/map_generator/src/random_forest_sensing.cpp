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
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT
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

// NEW: Include message header for publishing the percentage
#include <std_msgs/Float64.h> 

using namespace std;
using namespace Eigen;

ros::Publisher _all_map_pub;
ros::Publisher _obstacle_percentage_pub; // NEW: Publisher for the percentage
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
float ramdom_ratio = 0.5;
void RandomBRRTGenerate_Large(double size = 4)
{
   pcl::PointXYZ pt_random;
   random_device rd;
   default_random_engine eng(rd());
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
            for (float k = -1; k < _h_h; k += 0.5)
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
   std::cout << "size of map" << _x_l << " " << _x_h << " " << _y_l << " " << _y_h << " " << _h_h << std::endl;
   for (float i = _x_l; i < _x_h; i += 0.5)
   {

      for (float j = _y_l; j < _y_h; j += 0.5)
      {
         // get a random number between 0 and 1
         float random_num = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
         if (random_num < ramdom_ratio)
            for (float k = -1; k < _h_h; k += 0.5)
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

/**
 * @brief Helper function to add a wall of obstacle points to the global cloudMap.
 *
 * @param x1 Start x-coordinate of the wall.
 * @param y1 Start y-coordinate of the wall.
 * @param x2 End x-coordinate of the wall.
 * @param y2 End y-coordinate of the wall.
 * @param z_min Minimum height of the wall.
 * @param z_max Maximum height of the wall.
 * @param resolution Resolution for sampling points along the wall.
 */
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

/**
 * @brief Generates a fixed map with L, H, and U shaped obstacles.
 */
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
   // NEW: Advertise the percentage topic as latched
   _obstacle_percentage_pub = n.advertise<std_msgs::Float64>("obstacle_percentage", 1, true);

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
      FixedMapGenerate();
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
   else
   {
      ROS_ERROR("Unknown map type: '%s'. Defaulting to 'random_large'.", map_type.c_str());
      RandomBRRTGenerate_Large();
   }
   // =================================
   double obstacle_percentage = 0.0;
   if(map_type == "random_large"){
      obstacle_percentage = ramdom_ratio * 100.0;
      ROS_INFO("  Obstacle Percentage: %.2f %%", obstacle_percentage);
   }
   else {
         // === NEW: Calculate and Publish Obstacle Percentage ===
         ROS_INFO("Calculating obstacle percentage...");
         double total_volume = _x_size * _y_size * _z_size;
         double obstacle_points = static_cast<double>(cloudMap.points.size());
         double single_voxel_volume = pow(_resolution, 3);
         double obstacle_volume = obstacle_points * single_voxel_volume;
         
         if (total_volume > 1e-6) // Avoid division by zero
         {
            obstacle_percentage = (obstacle_volume / total_volume) * 100.0;
         }

         ROS_INFO("-----------------------------------------");
         ROS_INFO("Map Statistics:");
         ROS_INFO("  Total Map Volume (m^3): %.2f", total_volume);
         ROS_INFO("  Obstacle Points Count: %d", (int)obstacle_points);
         ROS_INFO("  Voxel Resolution (m): %.3f", _resolution);
         ROS_INFO("  Estimated Obstacle Volume (m^3): %.2f", obstacle_volume);
         ROS_INFO("-----------------------------------------");
   }

   std_msgs::Float64 percentage_msg;
   percentage_msg.data = obstacle_percentage;
   _obstacle_percentage_pub.publish(percentage_msg);
   ROS_INFO("Obstacle percentage published to latched topic: %s", _obstacle_percentage_pub.getTopic().c_str());
   // ==================================================
   

   // only pub map pointcloud on request
   ros::ServiceServer pub_glb_obs_service = n.advertiseService("/pub_glb_obs", pubGlbObs);
   ROS_INFO("Map generation complete. Ready to publish map on service call.");
   ros::spin();

   // ros::Rate loop_rate(_sense_rate);
   // while (ros::ok())
   // {
   //    ros::spinOnce();
   //    loop_rate.sleep();
   // }
}

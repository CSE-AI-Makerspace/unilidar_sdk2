/**********************************************************************
 Copyright (c) 2020-2024, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/

#include "unitree_lidar_sdk.h"

#include <csignal>
#include <iostream>
#include <memory>
#include <thread>

using namespace unilidar_sdk2;

namespace {
volatile std::sig_atomic_t running = 1;

void handleSignal(int) {
    running = 0;
}
}  // namespace

int main() {
    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);

    std::unique_ptr<UnitreeLidarReader> reader(createUnitreeLidarReader());

    const std::string lidar_ip = "192.168.1.62";
    const std::string local_ip = "192.168.1.2";
    const unsigned short lidar_port = 6101;
    const unsigned short local_port = 6201;

    if (reader->initializeUDP(lidar_port, lidar_ip, local_port, local_ip)) {
        std::cerr << "ERROR initializeUDP failed" << std::endl;
        return 1;
    }

    std::cerr << "Unitree L2 UDP initialized" << std::endl;
    reader->startLidarRotation();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    reader->setLidarWorkMode(0);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    reader->resetLidar();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    reader->startLidarRotation();

    PointCloudUnitree cloud;
    LidarImuData imu;
    uint64_t cloud_count = 0;
    uint64_t imu_count = 0;

    while (running) {
        const int result = reader->runParse();

        if (result == LIDAR_POINT_DATA_PACKET_TYPE && reader->getPointCloud(cloud)) {
            std::cout << "FRAME," << cloud_count++ << "," << cloud.stamp << ","
                      << cloud.points.size() << "," << cloud.ringNum << "\n";
            for (const auto &point : cloud.points) {
                std::cout << point.x << "," << point.y << "," << point.z << ","
                          << point.intensity << "," << point.ring << "\n";
            }
            std::cout << "END\n" << std::flush;
        } else if (result == LIDAR_IMU_DATA_PACKET_TYPE && reader->getImuData(imu)) {
            if ((++imu_count % 250) == 0) {
                std::cerr << "IMU packets parsed: " << imu_count << std::endl;
            }
        } else if (result == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    reader->stopLidarRotation();
    reader->closeUDP();
    std::cerr << "Unitree L2 UDP stopped" << std::endl;
    return 0;
}

#include <cstdint>
#include <vector>
#include <string>
#include <limits>
#include <chrono>
#include <cmath>
#include <array>

//#include <ammintrin.h>
#include <immintrin.h>

#include <iostream>
#include <fstream>
#include <sstream>

typedef float PointCoord;

class Timer {
	std::string name;
	std::chrono::system_clock::time_point start;

public:
	Timer(std::string name) : name(name), start(std::chrono::system_clock::now()) {
	}

	~Timer() {
		auto end=std::chrono::system_clock::now();
		std::cout<<name<<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()<<std::endl;
	}
};

struct Point {
	PointCoord x;
	PointCoord y;

	Point() {}
	Point(PointCoord x, PointCoord y) : x(x),y(y) {
	}
};

std::vector<std::string> readFile(std::string fileName) {
	std::vector<std::string> lines;
	std::ifstream file(fileName);
	std::string line;
	while (std::getline(file, line)) {
	   lines.push_back(line);
	}

	return lines;
}

std::vector<Point> parsePoints(std::vector<std::string>& lines) {
	std::vector<Point> points;
	points.resize(lines.size());

	const int numLines=lines.size();

	for (int i = 0; i < numLines; ++i) {
		std::string part;
		std::string line = lines[i];
		std::stringstream ss(line);
		std::getline(ss, part, ',');
		PointCoord x = std::stod(part);
		std::getline(ss, part, ',');
		PointCoord y = std::stod(part);
		points[i]=Point(x,y);
	}
	return points;
}

std::vector<Point> parseCenters(std::vector<std::string>& lines) {
	std::vector<Point> points;
	points.resize(lines.size());
	
	const int numLines=lines.size();

	for (int i = 0; i < numLines; ++i) {
		std::string part;
		std::string line = lines[i];
		std::stringstream ss(line);
		std::getline(ss, part, ',');
		std::getline(ss, part, ',');
		PointCoord x = std::stod(part);
		std::getline(ss, part, ',');
		PointCoord y = std::stod(part);
		points[i]=Point(x,y);
	}
	return points;
}

struct CenterData {
	PointCoord xSum;
	PointCoord ySum;
	uint64_t count;

	CenterData() : xSum(0), ySum(0), count(0) {
	}

	Point center() {
		return Point(xSum/count, ySum/count);
	}
};

template<int NumClusters> std::vector<Point> kmeans_naive(std::array<Point,NumClusters> pcenters, const std::vector<Point> points, const uint32_t numIterations) {
	const size_t numCenters = pcenters.size();
	const int numPoints=points.size();
	std::array<Point,NumClusters> centers = pcenters;

	for (unsigned i = 0; i < numIterations; ++i) {
		Timer time("Iteration ");
		std::array<CenterData,NumClusters> centerAggregates;

		for (int j = 0; j < numPoints; ++j) {
			Point point = points[j];
			PointCoord bestError=std::numeric_limits<PointCoord>::max();
			auto bestCenter = 0;

			// Find nearest center
			for (unsigned k = 0; k < numCenters; ++k) {
				const Point center = centers[k];
				auto distance=sqrt((point.x-center.x)*(point.x-center.x)+(point.y-center.y)*(point.y-center.y));
				if(distance<bestError) {
					bestError=distance;
					bestCenter=k;
				}
			}

			// Assign point to center
			centerAggregates[bestCenter].xSum += point.x;
			centerAggregates[bestCenter].ySum += point.y;
			centerAggregates[bestCenter].count++;
		}

		// Recompute centers
		for (unsigned j = 0; j < centers.size(); ++j) {
			centers[j] = centerAggregates[j].center();
		}
	}

	return std::vector<Point>(centers.begin(),centers.end());
}

template<int NumClusters> std::vector<Point> kmeans_b(std::array<Point,NumClusters> pcenters, const std::vector<Point> points, const uint32_t numIterations) {
	const size_t numCenters = pcenters.size();
	const int numPoints=points.size();
	alignas(512) std::array<Point,NumClusters> centers = pcenters;
	std::vector<uint8_t> bestCenters;
	bestCenters.resize(points.size());

	for (unsigned i = 0; i < numIterations; ++i) {
		Timer time("Iteration ");
		std::array<CenterData,NumClusters> centerAggregates;

		for (int j = 0; j < numPoints; ++j) {
			// Compute distnce to each cluster
			alignas(512) std::array<PointCoord,NumClusters> distances;
			{
				const Point point=points[j];
				for (unsigned k = 0; k < numCenters; ++k) {
					const Point center = centers[k];
					distances[k]=(point.x-center.x)*(point.x-center.x)+(point.y-center.y)*(point.y-center.y);
				}
			}

			// Find closest center
			{
				auto bestCenter=0;
				for (unsigned i = 0; i < numCenters; ++i) {
					if(distances[i]<distances[bestCenter]) {
						bestCenter=i;
					}
				}
				bestCenters[j]=bestCenter;
			}
		}

		for (int j = 0; j < numPoints; ++j) {
			uint8_t center = bestCenters[j];
			Point point = points[j];
			centerAggregates[center].xSum += point.x;
			centerAggregates[center].ySum += point.y;
			centerAggregates[center].count++;
		}

		for (unsigned j = 0; j < centers.size(); ++j) {
			centers[j] = centerAggregates[j].center();
		}
	}

	return std::vector<Point>(centers.begin(),centers.end());
}

template<int NumClusters> std::vector<Point> kmeans_a(std::array<Point,NumClusters> pcenters, const std::vector<Point> points, const uint32_t numIterations) {
	const size_t numCenters = pcenters.size();
	const int numPoints=points.size();
	alignas(512) std::array<Point,NumClusters> centers = pcenters;
	std::vector<uint8_t> bestCenters;
	bestCenters.resize(points.size());

	for (unsigned i = 0; i < numIterations; ++i) {
		Timer time("Iteration ");
		std::array<CenterData,NumClusters> centerAggregates;

		for (int j = 0; j < numPoints; ++j) {
			// Compute distnce to each cluster
			alignas(512) std::array<PointCoord,NumClusters> distances;
			{
				const Point point=points[j];
				for (unsigned k = 0; k < numCenters; ++k) {
					const Point center = centers[k];
					distances[k]=(point.x-center.x)*(point.x-center.x)+(point.y-center.y)*(point.y-center.y);
				}
			}

			// Find closest center
			{
				auto bestCenter=0;
				PointCoord bestError=std::numeric_limits<PointCoord>::max();
				for (unsigned k = 0; k < numCenters; ++k) {
					if(bestError>distances[k]) {
						bestError=distances[k];
					}
				}

				for (unsigned k = 0; k < numCenters; ++k) {
					if(bestError==distances[k]) {
						bestCenter=k;
					}
				}
				bestCenters[j]=bestCenter;
			}
		}

		for (int j = 0; j < numPoints; ++j) {
			uint8_t center = bestCenters[j];
			Point point = points[j];
			centerAggregates[center].xSum += point.x;
			centerAggregates[center].ySum += point.y;
			centerAggregates[center].count++;
		}

		for (unsigned j = 0; j < centers.size(); ++j) {
			centers[j] = centerAggregates[j].center();
		}
	}

	return std::vector<Point>(centers.begin(),centers.end());
}

int main(int argc, char** argv) {
	if(argc!=3) {
		std::cout<<"Missing parameters, expecting ./cluster <centers_file> <points_file>"<<std::endl;
		throw;
	}

	auto centerLines=readFile(std::string(argv[1]));
	auto centers=parseCenters(centerLines);

	auto pointLines=readFile(std::string(argv[2]));
	auto points=parsePoints(pointLines);

	std::cout<<"Num centers: "<<centers.size()<<std::endl;
	std::cout<<"Num points: "<<points.size()<<std::endl;

	static const int numCluster=32;
	std::array<Point,numCluster> clusterCenters;
	for (int i = 0; i < numCluster; ++i) {
		clusterCenters[i]=centers[i];
	}

	{
		Timer time("Total Naive ");
		auto clusters=kmeans_b<numCluster>(clusterCenters,points,20);
		for(auto& center: clusters) {
			std::cout<<center.x<<", "<<center.y<<std::endl;
		}
	}

	{
		Timer time("Total Opt A ");
		auto clusters=kmeans_a<numCluster>(clusterCenters,points,20);
		for(auto& center: clusters) {
			std::cout<<center.x<<", "<<center.y<<std::endl;
		}
	}

	{
		Timer time("Total Opt B ");
		auto clusters=kmeans_b<numCluster>(clusterCenters,points,20);
		for(auto& center: clusters) {
			std::cout<<center.x<<", "<<center.y<<std::endl;
		}
	}

	return 0;
}

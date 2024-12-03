/*
yolov8 segmentation simplest demo using opencv dnn and onnxruntime
winxos 20230607
c++版本选择旧14会有问题，选择c++17
*/
#include<iostream>
#include<memory>
#include <chrono>

#include <vector>
#include <cmath>
#include <limits>


#include "npy/core.h"
#include "npy/tensor.h"
#include "npy/npy.h"
#include "npy/npz.h"

using namespace std;

#include<opencv2/opencv.hpp>
using namespace cv;
using namespace cv::dnn;

#include<onnxruntime_cxx_api.h>
using namespace Ort;


Ort::Env *env;
Session* session;
Ort::SessionOptions *sessionOptions;


npy::tensor<std::float_t> current_positions(std::vector<size_t> {1730, 6, 2});
npy::tensor<std::float_t> node_features(std::vector<size_t> {1730, 30});
npy::tensor<std::int64_t> edge_index(std::vector<size_t> {2, 15820});
npy::tensor<std::float_t> edge_features(std::vector<size_t> {15820, 3});


float squared_distance(const std::vector<float>& a, const std::vector<float>& b);
std::pair<std::vector<int64>, std::vector<int64>> radius_graph(const std::vector<std::vector<float>>& node_features, float r, const std::vector<int>& batch_ids, bool loop = false, int max_num_neighbors = 128);
void encoder_preprocess(void);
int radius_graph_test(void);
bool gns_ort();



int main(void)
{
	/*
	* 注意python中有些int是int64，在这里int是32位，int64才是64位8字节
	*/
	srand(time(0));
	cout << "int:" << sizeof(int) << "float:" << sizeof(float_t)  <<  " int64_t:" << sizeof(int64_t) << endl;
	gns_ort();
	//radius_graph_test();//没有输出，表示能够正确替代radius_graph这个函数
	std::cout << "Press Enter to exit...";
	std::cin.get(); // 等待用户输入
	return 0;
}

// 计算欧几里得距离的平方
float squared_distance(const std::vector<float>& a, const std::vector<float>& b)
{
	float sum = 0.0;
	for (size_t i = 0; i < a.size(); ++i) {
		sum += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return sum;
}

// 计算半径图的边连接
std::pair<std::vector<int64>, std::vector<int64>> radius_graph(
	const std::vector<std::vector<float>>& node_features, // 每个节点的位置特征
	float r, // 半径
	const std::vector<int>& batch_ids, // 每个节点的批次ID
	bool loop , // 是否包含自环
	int max_num_neighbors ) { // 每个节点最多的邻居数

	size_t nparticles = node_features.size();
	size_t num_dimensions = node_features[0].size(); // 假设每个节点的维度相同
	std::vector<int64> senders;
	std::vector<int64> receivers;

	// 循环遍历每一对节点，检查它们之间的距离
	for (size_t i = 0; i < nparticles; ++i) {
		for (size_t j = 0; j < nparticles; ++j) {
			// 跳过自己与自己的连接（如果不添加自环）
			if (!loop && i == j) continue;

			// 计算节点i和节点j之间的距离
			float dist_sq = squared_distance(node_features[i], node_features[j]);

			// 如果距离小于r的平方，表示这两个节点是邻居
			if (dist_sq < r * r) {
				senders.push_back(i);
				receivers.push_back(j);
			}
		}
	}

	// 限制每个节点的最大邻居数
	if (max_num_neighbors > 0) {
		std::vector<int64> node_neighbor_count(nparticles, 0);
		std::vector<int64> filtered_senders;
		std::vector<int64> filtered_receivers;

		// 遍历发送者列表，根据最大邻居数过滤
		for (size_t k = 0; k < senders.size(); ++k) {
			int64 sender = senders[k];
			if (node_neighbor_count[sender] < max_num_neighbors) {
				filtered_senders.push_back(senders[k]);
				filtered_receivers.push_back(receivers[k]);
				node_neighbor_count[sender]++;
			}
		}

		senders = filtered_senders;
		receivers = filtered_receivers;
	}

	return { senders, receivers };
}

void encoder_preprocess(void)
{/*函数功能：读取current_positions.npy，node_features.npy，edge_features.npy数据，并调用radius_graph函数生成edge_index*/
	//read npz
	npy::inpzstream input("gns_input_data.npz");//放在最外面工程文件坐在目录//"current_positions","node_features","edge_index","edge_features"
	
	current_positions = input.read<std::float_t>("current_positions.npy");
	node_features = input.read<std::float_t>("node_features.npy");
	//edge_index = input.read<std::int64_t>("edge_index.npy");//由radius_graph函数生成
	edge_features = input.read<std::float_t>("edge_features.npy");

	//由node_feature生成edge_index begin
	vector<vector<float>> node_features_vector;
	for (int i = 0; i < 1730; ++i) {// 遍历张量的行,将numpy转换为vector
		std::vector<float> rowVec;
		for (int j = 0; j < 2; ++j) // 遍历当前行的列元素
			rowVec.push_back(node_features(i, j));
		node_features_vector.push_back(rowVec);
	}
	std::vector<int> batch_ids(1730, 0);
	float radius = 0.015;// 半径 (邻居的最大距离)
	bool add_self_edges = true;// 是否允许自环
	int max_neighbors = 128;// 最大邻居数

	// 调用 radius_graph 函数
	std::pair<std::vector<int64>, std::vector<int64>> result = radius_graph(node_features_vector, radius, batch_ids, add_self_edges, max_neighbors);

	vector<int64> sender = result.first;
	vector<int64> receiver = result.second;
	for (int i = 0; i < 15820; i++)
	{
		edge_index(0, i) = receiver[i];
		edge_index(1, i) = sender[i];
	}
}


int radius_graph_test(void)
{//测试函数能否成功使用radius_graph
	
	npy::inpzstream input("radius_graph_data.npz");

	npy::tensor<std::float_t> node_features(std::vector<size_t> {1730, 2});
	npy::tensor<std::int64_t> edge_index(std::vector<size_t> {2, 15820});

	node_features = input.read<std::float_t>("node_features.npy");
	edge_index = input.read<std::int64_t>("edge_index.npy");

	vector<vector<float>> node_features_vector;

	for (int i = 0; i < 1730; ++i) {// 遍历张量的行,将numpy转换为vector
		std::vector<float> rowVec;
		for (int j = 0; j < 2; ++j) // 遍历当前行的列元素
			rowVec.push_back(node_features(i, j));
		node_features_vector.push_back(rowVec);
	}

	vector<vector<int64_t>> edge_index_vector;
	for (int i = 0; i < 2; ++i) {// 遍历张量的行
		std::vector<int64_t> rowVec;
		for (int j = 0; j < 15820; ++j) // 遍历当前行的列元素
			rowVec.push_back(edge_index(i, j));
		edge_index_vector.push_back(rowVec);
	}

	std::vector<int> batch_ids(1730, 0);
	float radius = 0.015;// 半径 (邻居的最大距离)
	bool add_self_edges = true;// 是否允许自环
	int max_neighbors = 128;// 最大邻居数

	// 调用 radius_graph 函数
	std::pair<std::vector<int64>, std::vector<int64>> result = radius_graph(node_features_vector, radius, batch_ids, add_self_edges, max_neighbors);


	vector<int64> sender = result.first;
	vector<int64> receiver = result.second;

	for (int i = 0; i < 15820; i++)
	{
		if (receiver[i] != edge_index_vector[0][i] || sender[i] != edge_index_vector[1][i])
		{
			cout << i << " " << "true:" << edge_index_vector[0][i] << "-" << edge_index_vector[1][i] << "     ";
			cout << "pred:" << sender[i] << "-" << receiver[i] << endl;

		}
	}
	std::cout << std::endl;
	return 0;
}

bool gns_ort()
{
	encoder_preprocess();
	

	//由node_feature生成edge_index end
	Ort::Value input_tensor_current_positions = Value::CreateTensor<float>(MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
		(float*)current_positions.data(), 1730 * 6 * 2, vector<int64_t>{ 1730, 6, 2 }.data(), 3);//参数为const OrtMemoryInfo* info, T* p_data, size_t p_data_element_count, const int64_t* shape, size_t shape_len
	Ort::Value input_tensor_node_features = Value::CreateTensor<float_t>(MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
		(float_t*)node_features.data(), 1730 * 30, vector<int64_t>{ 1730, 30 }.data(), 2);
	Ort::Value input_tensor_edge_index = Value::CreateTensor<int64_t>(MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
		(int64_t*)edge_index.data(), 2 * 15820, vector<int64_t>{2, 15820}.data(), 2);
	Ort::Value input_tensor_edge_features = Value::CreateTensor<float>(MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
		(float*)edge_features.data(), 15820 * 3, vector<int64_t>{ 15820, 3}.data(), 2);




	/*// 打印数据
	size_t num_elemen = 10 * 260 * 130; // 10 * 260 * 130
	// 设置格式
	std::cout << std::fixed << std::setprecision(2);
	for (size_t i = num_elemen-10000; i < num_elemen; ++i) {
		std::cout << data_ptr[i] << "-"; // 打印每个元素
	}*/

	std::vector<Ort::Value> input_tensor;
	input_tensor.push_back(std::move(input_tensor_current_positions));
	input_tensor.push_back(std::move(input_tensor_node_features));
	input_tensor.push_back(std::move(input_tensor_edge_index));
	input_tensor.push_back(std::move(input_tensor_edge_features));


	sessionOptions = new Ort::SessionOptions;
	env = new Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "default");
	//sessionOptions->SetIntraOpNumThreads(2);//启用所有可能的优化。
	//sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	session = new Ort::Session(*env, L"gns.onnx", *sessionOptions);


	vector<const char*> input_names = { "current_positions","node_features","edge_index","edge_features" };
	vector<const char*> output_names = { "next_position" };
	const char** names = input_names.data();
	cout << "names:" << endl;
	for (int i = 0; i < 4; i++)
		cout << i << ": " << names[i] << endl;
	auto start = chrono::high_resolution_clock::now();
	int nsteps = 1;


	auto outputs = session->Run(RunOptions{ nullptr }, input_names.data(),
		const_cast<Value*>(input_tensor.data()), 4, const_cast<char**>(output_names.data()), 1);

	auto end = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
	cout << "ort time: " << duration << " millis.";

	float* all_data = outputs[0].GetTensorMutableData<float>();
	auto data_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();//1*44*8400//检测框
	std::cout << "Press Enter to exit...";
	std::cin.get(); // 等待用户输入
	return 1;
}

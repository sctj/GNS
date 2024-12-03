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

//#include<format>
using namespace std;

#include<opencv2/opencv.hpp>
using namespace cv;
using namespace cv::dnn;

#include<onnxruntime_cxx_api.h>
using namespace Ort;

struct Obj {
	int id = 0;
	float accu = 0.0;
	Rect bound;
	Mat mask;
};
int seg_ch = 32;
int seg_w = 160, seg_h = 160;
int net_w = 640, net_h = 640;
float accu_thresh = 0.25, mask_thresh = 0.5;
Ort::Env *env;
Session* session;
Ort::SessionOptions *sessionOptions;
struct ImageInfo {
	Size raw_size;
	Vec4d trans;
};
vector<string> class_names = { "bean","corn","impurity","rice","ricebrown","ricemilled","wheat","xian" };




bool gns_ort(string& mpath)
{
	//read npz
	npy::inpzstream input("gns_input_data.npz");//放在最外面工程文件坐在目录//"current_positions","node_features","edge_index","edge_features"
	cout << "float:" << sizeof(float_t) << " int64_t:" << sizeof(int64_t) << endl;
	npy::tensor<std::float_t> current_positions(std::vector<size_t> {1730,6, 2});
	npy::tensor<std::float_t> node_features(std::vector<size_t> {1730,30});
	npy::tensor<std::int64_t> edge_index(std::vector<size_t> {2, 15820});
	npy::tensor<std::float_t> edge_features(std::vector<size_t> {15820, 3});

	

	current_positions = input.read<std::float_t>("current_positions.npy");
	node_features = input.read<std::float_t>("node_features.npy");
	edge_index = input.read<std::int64_t>("edge_index.npy");
	edge_features = input.read<std::float_t>("edge_features.npy");

	


	
	Ort::Value input_tensor_current_positions = Value::CreateTensor<float>(MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
		(float*)current_positions.data(), 1730* 6*2,vector<int64_t>{ 1730, 6, 2 }.data(), 3);//参数为const OrtMemoryInfo* info, T* p_data, size_t p_data_element_count, const int64_t* shape, size_t shape_len
	Ort::Value input_tensor_node_features = Value::CreateTensor<float_t>(MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
		(float_t*)node_features.data(), 1730* 30, vector<int64_t>{ 1730, 30 }.data(), 2);
	Ort::Value input_tensor_edge_index = Value::CreateTensor<int64_t>(MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
		(int64_t*)edge_index.data(), 2*15820, vector<int64_t>{2, 15820}.data(), 2);
	Ort::Value input_tensor_edge_features = Value::CreateTensor<float>(MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
		(float*)edge_features.data(), 15820*3, vector<int64_t>{ 15820, 3}.data(), 2);
	
	


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
	const char** names =input_names.data();
	cout << "names:" << endl;
	for (int i = 0; i < 4; i++)
		cout << i << ": " << names[i] << endl;
	auto start = chrono::high_resolution_clock::now();
	int nsteps = 1;
	//for (int step = 0; step < nsteps; step++)
	//{

	

	auto outputs = session->Run(RunOptions{ nullptr },input_names.data(), 
		const_cast<Value*>(input_tensor.data()), 4,const_cast<char**>(output_names.data()), 1);



//	}
	auto end = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
	cout << "ort time: " << duration << " millis.";
	
	float* all_data = outputs[0].GetTensorMutableData<float>();
	auto data_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();//1*44*8400//检测框
	std::cout << "Press Enter to exit...";
    std::cin.get(); // 等待用户输入
	return 1;
}

float squared_distance(const std::vector<float>& a, const std::vector<float>& b);
std::pair<std::vector<int64>, std::vector<int64>> radius_graph(const std::vector<std::vector<float>>& node_features, float r, const std::vector<int>& batch_ids, bool loop = false,int max_num_neighbors = 128);
int radius_graph_test(void)
{
	//read npz
	npy::inpzstream input("radius_graph_data.npz");//测试函数能否成功使用radius_graph
	
	npy::tensor<std::float_t> node_features(std::vector<size_t> {1730, 2});
	npy::tensor<std::int64_t> edge_index(std::vector<size_t> {2, 15820});

	node_features = input.read<std::float_t>("node_features.npy");
	edge_index = input.read<std::int64_t>("edge_index.npy");

	vector<vector<float>> node_features_vector;
	
	for (int i = 0; i < 1730; ++i) {// 遍历张量的行,将numpy转换为vector
		std::vector<float> rowVec;
		for (int j = 0; j <2; ++j) // 遍历当前行的列元素
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

	std::vector<int> batch_ids(1730,0);
	float radius = 0.015;// 半径 (邻居的最大距离)
	bool add_self_edges = true;// 是否允许自环
	int max_neighbors = 128;// 最大邻居数

	// 调用 radius_graph 函数
	std::pair<std::vector<int64>, std::vector<int64>> result = radius_graph(node_features_vector, radius, batch_ids, add_self_edges, max_neighbors);


	vector<int64> sender = result.first;
	vector<int64> receiver = result.second;

	for (int i = 0; i < 15820; i++)
	{
		if ( receiver[i] != edge_index_vector[0][i] || sender[i] != edge_index_vector[1][i] )
		{
			cout << i<< " "<<"true:" << edge_index_vector[0][i] << "-" << edge_index_vector[1][i] << "     ";
			cout << "pred:" << sender[i] << "-" << receiver[i] << endl;
			
		}
	}
	std::cout << std::endl;
	return 0;
}
	
void encoder_preprocess(void)
{
	//周一实现该函数，再迭代推理
}
int main()
{
	/*
	* 注意python中有些int是int64，在这里int是32位，int64才是64位8字节
	*/
	srand(time(0));
	cout << "int:" << sizeof(int) << " int64_t:" << sizeof(int64_t) << endl;
	string model_path = "./gns.onnx";
	//Mat img2 = img1.clone();
	//vector<Obj> result;
	//如果需要dnn推理则取消注释，dnn推理时间2900ms，ort为530ms
	//gns_ort(model_path);
	radius_graph_test();//没有输出，表示能够正确替代radius_graph这个函数
	
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






/*从模型中读出输入名称，但是输出发现部分出现乱码
* 	int num_input_nodes = 4;
	vector<const char*> input_node_names(num_input_nodes);
	vector<int64_t> input_node_dims;
	Ort::AllocatorWithDefaultOptions  allocator ;
	for (int i = 0; i < num_input_nodes; i++) {
		AllocatedStringPtr input_name1 = session->GetInputNameAllocated(i, allocator);
		input_node_names[i] = input_name1.get();

		Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
		ONNXTensorElementDataType type = tensor_info.GetElementType();
		input_node_dims = tensor_info.GetShape();
		cout << "i:" << i << " name:" << input_name1.get() << " type:" << tensor_info << " dims:";
			for (int i = 0; i < input_node_dims.size(); i++)
			{
				cout << input_node_dims[i] << " ";
			}
		cout << endl;
	}
	for (int i = 0; i < 4; i++)
		cout << i << ": " << input_node_names[i] <<endl;

void decode_output(Mat& output0, Mat& output1, ImageInfo para, vector<Obj>& output)
{
	output.clear();
	vector<int> class_ids;
	vector<float> accus;
	vector<Rect> boxes;
	vector<vector<float>> masks;
	int data_width = class_names.size() + 4 + 32;
	int rows = output0.rows;
	float* pdata = (float*)output0.data;
	for (int r = 0; r < rows; ++r)
	{
		Mat scores(1, class_names.size(), CV_32FC1, pdata + 4);
		Point class_id;
		double max_socre;
		minMaxLoc(scores, 0, &max_socre, 0, &class_id);
		if (max_socre >= accu_thresh)
		{
			masks.push_back(vector<float>(pdata + 4 + class_names.size(), pdata + data_width));
			float w = pdata[2] / para.trans[0];
			float h = pdata[3] / para.trans[1];
			int left = MAX(int((pdata[0] - para.trans[2]) / para.trans[0] - 0.5 * w + 0.5), 0);
			int top = MAX(int((pdata[1] - para.trans[3]) / para.trans[1] - 0.5 * h + 0.5), 0);
			class_ids.push_back(class_id.x);
			accus.push_back(max_socre);
			boxes.push_back(Rect(left, top, int(w + 0.5), int(h + 0.5)));
		}
		pdata += data_width;//next line
	}
	vector<int> nms_result;
	NMSBoxes(boxes, accus, accu_thresh, mask_thresh, nms_result);//opencv 内置NMSBoxes
	for (int i = 0; i < nms_result.size(); ++i)
	{
		int idx = nms_result[i];
		boxes[idx] = boxes[idx] & Rect(0, 0, para.raw_size.width, para.raw_size.height);
		Obj result = { class_ids[idx] ,accus[idx] ,boxes[idx] };
		get_mask(Mat(masks[idx]).t(), output1, para, boxes[idx], result.mask);
		output.push_back(result);
	}
}

		
		
		
void get_mask(const Mat& mask_info, const Mat& mask_data, const ImageInfo& para, Rect bound, Mat& mast_out)
{
	Vec4f trans = para.trans;
	int r_x = floor((bound.x * trans[0] + trans[2]) / net_w * seg_w);
	int r_y = floor((bound.y * trans[1] + trans[3]) / net_h * seg_h);
	int r_w = ceil(((bound.x + bound.width) * trans[0] + trans[2]) / net_w * seg_w) - r_x;
	int r_h = ceil(((bound.y + bound.height) * trans[1] + trans[3]) / net_h * seg_h) - r_y;
	r_w = MAX(r_w, 1);
	r_h = MAX(r_h, 1);
	if (r_x + r_w > seg_w) //crop
	{
		seg_w - r_x > 0 ? r_w = seg_w - r_x : r_x -= 1;
	}
	if (r_y + r_h > seg_h)
	{
		seg_h - r_y > 0 ? r_h = seg_h - r_y : r_y -= 1;
	}
	vector<Range> roi_rangs = { Range(0, 1) ,Range::all() , Range(r_y, r_h + r_y) ,Range(r_x, r_w + r_x) };
	Mat temp_mask = mask_data(roi_rangs).clone();
	Mat protos = temp_mask.reshape(0, { seg_ch,r_w * r_h });
	Mat matmul_res = (mask_info * protos).t();
	Mat masks_feature = matmul_res.reshape(1, { r_h,r_w });
	Mat dest;
	exp(-masks_feature, dest);//sigmoid
	dest = 1.0 / (1.0 + dest);
	int left = floor((net_w / seg_w * r_x - trans[2]) / trans[0]);
	int top = floor((net_h / seg_h * r_y - trans[3]) / trans[1]);
	int width = ceil(net_w / seg_w * r_w / trans[0]);
	int height = ceil(net_h / seg_h * r_h / trans[1]);
	Mat mask;
	resize(dest, mask, Size(width, height));
	mast_out = mask(bound - Point(left, top)) > mask_thresh;
}

void draw_result(Mat img, vector<Obj>& result, vector<Scalar> color)
{
	Mat mask = img.clone();
	for (int i = 0; i < result.size(); i++)
	{
		int left, top;
		left = result[i].bound.x;
		top = result[i].bound.y;
		int color_num = i;
		rectangle(img, result[i].bound, color[result[i].id], 8);
		if (result[i].mask.rows && result[i].mask.cols > 0)
		{
			mask(result[i].bound).setTo(color[result[i].id], result[i].mask);
		}
		string label = std::format("{}:{:.2f}", class_names[result[i].id], result[i].accu);
		putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 2, color[result[i].id], 4);
	}
	addWeighted(img, 0.6, mask, 0.4, 0, img); //add mask to src
	resize(img, img, Size(640, 640));
	imshow("img", img);
	waitKey();
}
*/

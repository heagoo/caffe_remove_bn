#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;
using std::string;
using std::vector;

static bool startWith(const string &str, const string &strStart) {
  return str.compare(0, strStart.size(), strStart) == 0 ? true : false;
}

static bool isNeededLayer(const vector<string> &lines) {
  string name;
  string type;
  string bottom;
  string top;

  const string strName("name: \"");
  const string strType("type: \"");
  const string strBottom("bottom: \"");
  const string strTop("top: \"");

  for (int i = 0; i < lines.size(); ++i) {
    size_t pos = lines[i].find(strName);
    if (pos != string::npos) {
      size_t start = pos + strName.length();
      size_t end = lines[i].find('\"', start);
      if (end != string::npos) {
        name = lines[i].substr(start, end - start);
        continue;
      }
    }

    pos = lines[i].find(strType);
    if (pos != string::npos) {
      size_t start = pos + strType.length();
      size_t end = lines[i].find('\"', start);
      if (end != string::npos) {
        type = lines[i].substr(start, end - start);
        continue;
      }
    }

    pos = lines[i].find(strBottom);
    if (pos != string::npos) {
      size_t start = pos + strBottom.length();
      size_t end = lines[i].find('\"', start);
      if (end != string::npos) {
        bottom = lines[i].substr(start, end - start);
        continue;
      }
    }

    pos = lines[i].find(strTop);
    if (pos != string::npos) {
      size_t start = pos + strTop.length();
      size_t end = lines[i].find('\"', start);
      if (end != string::npos) {
        top = lines[i].substr(start, end - start);
        continue;
      }
    }
  }
  
  bool ret = (type != "BatchNorm" && type != "Scale");

  // TODO: For current impl.,rRemoved layer MUST have the same top and bottom
  if (!ret && top != bottom) {
    std::cout << "NOT supported deploy file: layer " << name 
              << " has different top and bottom" << std::endl;
    exit(-1);
  }

  return ret;
}

static void saveToFile(std::ofstream &file, const vector<string> &lines) {
  bool is_conv_layer = false;
  const string target("bias_term: false");

  for (int i = 0; i < lines.size(); ++i) {
    string line = lines[i];
    if (line.find("type: \"Convolution\"") != string::npos) {
      is_conv_layer = true;
    }
    // Change "bias_term: false" to "bias_term: true"
    size_t pos = line.find(target);
    if (pos != string::npos) {
      line.replace(pos, target.length(), "bias_term: true");
    }
    file << line << std::endl;
  }
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " bn_removed.prototxt bn_removed.caffemodel" << std::endl;
    return 1;
  }

  string deploy_file   = argv[1];
  string trained_file = argv[2];
  string new_deploy_file = argv[3];
  string new_model_file = argv[4];

  // Remove the BatchNorm and Scale layer
  // TODO: lots of abnormal situations not considered in current impl.
  std::ifstream prototxt(deploy_file.c_str());
  std::ofstream newproto(new_deploy_file.c_str());
  CHECK(prototxt) << "Unable to open deploy file " << deploy_file;

  vector<string> layer_lines;
  string line;
  while (std::getline(prototxt, line)) {
    if (startWith(line, "layer {")) {
      // Save previous layer/content
      if (isNeededLayer(layer_lines)) {
        saveToFile(newproto, layer_lines);
      }
      layer_lines.clear();
    }
    layer_lines.push_back(line);
  }
  // Save the remaining
  saveToFile(newproto, layer_lines);
  newproto.close();

  // Transform the model file
  shared_ptr<Net<float> > net;
  net.reset(new Net<float>(new_deploy_file, TEST));
  net->TransformTrainedLayersFrom(trained_file);

  NetParameter net_param;
  net->ToProto(&net_param, false);
  WriteProtoToBinaryFile(net_param, new_model_file);
}

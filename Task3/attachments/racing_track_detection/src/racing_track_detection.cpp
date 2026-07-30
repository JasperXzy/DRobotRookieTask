#include "racing_track_detection/racing_track_detection.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <opencv2/opencv.hpp>

#include "dnn_node/util/image_proc.h"
#include "foxglove_msgs/msg/circle_annotation.hpp"
#include "foxglove_msgs/msg/text_annotation.hpp"
#include "geometry_msgs/msg/point32.hpp"
#include "hobot_cv/hobotcv_imgproc.h"
#include "rclcpp/qos.hpp"

namespace
{
constexpr int kModelWidth = 224;
constexpr int kModelHeight = 224;
constexpr float kControlWidth = 640.0F;
constexpr float kControlHeight = 480.0F;
constexpr float kRoiTopControl = 256.0F;

float Sigmoid(float value)
{
  const float clipped = std::max(-80.0F, std::min(80.0F, value));
  return 1.0F / (1.0F + std::exp(-clipped));
}

int EvenFloor(int value)
{
  return value - value % 2;
}
}  // namespace

TrackDetectionNode::TrackDetectionNode(
  const std::string & node_name,
  const rclcpp::NodeOptions & options)
: DnnNode(node_name, options)
{
  this->declare_parameter<std::string>("model_path", "");
  this->declare_parameter<std::string>("sub_img_topic", sub_img_topic_);
  this->declare_parameter<std::string>("output_topic", output_topic_);
  this->declare_parameter<std::string>("sign_topic", sign_topic_);
  this->declare_parameter<std::string>("camera_frame_id", camera_frame_id_);
  this->declare_parameter<std::string>("annotations_topic", annotations_topic_);
  this->declare_parameter<double>(
    "annotation_confidence_threshold",
    annotation_confidence_threshold_);
  this->declare_parameter<int>("bpu_core_id", bpu_core_id_);

  this->get_parameter("model_path", model_path_);
  this->get_parameter("sub_img_topic", sub_img_topic_);
  this->get_parameter("output_topic", output_topic_);
  this->get_parameter("sign_topic", sign_topic_);
  this->get_parameter("camera_frame_id", camera_frame_id_);
  this->get_parameter("annotations_topic", annotations_topic_);
  this->get_parameter(
    "annotation_confidence_threshold",
    annotation_confidence_threshold_);
  this->get_parameter("bpu_core_id", bpu_core_id_);

  if (model_path_.empty()) {
    RCLCPP_ERROR(get_logger(), "Parameter model_path must point to a generated X5 BIN.");
    throw std::runtime_error("model_path is empty");
  }
  if (bpu_core_id_ != 0 && bpu_core_id_ != 1) {
    RCLCPP_ERROR(get_logger(), "bpu_core_id must be 0 or 1, got %d", bpu_core_id_);
    throw std::runtime_error("invalid bpu_core_id");
  }
  if (Init() != 0) {
    RCLCPP_ERROR(get_logger(), "DNN initialization failed for %s", model_path_.c_str());
    throw std::runtime_error("DNN initialization failed");
  }
  initialized_ = true;

  publisher_ = this->create_publisher<ai_msgs::msg::PerceptionTargets>(
    output_topic_,
    rclcpp::SensorDataQoS());
  annotations_publisher_ =
    this->create_publisher<foxglove_msgs::msg::ImageAnnotations>(
    annotations_topic_,
    rclcpp::SensorDataQoS());
  image_subscriber_ =
    this->create_subscription<hbm_img_msgs::msg::HbmMsg1080P>(
    sub_img_topic_,
    rclcpp::SensorDataQoS(),
    std::bind(&TrackDetectionNode::ImageCallback, this, std::placeholders::_1));
  sign_subscriber_ = this->create_subscription<std_msgs::msg::Int32>(
    sign_topic_,
    rclcpp::SensorDataQoS(),
    std::bind(&TrackDetectionNode::SignCallback, this, std::placeholders::_1));

  RCLCPP_INFO(
    get_logger(),
    "Ready: model=%s input=%s output=%s annotations=%s core=%d",
    model_path_.c_str(),
    sub_img_topic_.c_str(),
    output_topic_.c_str(),
    annotations_topic_.c_str(),
    bpu_core_id_);
}

void TrackDetectionNode::SignCallback(const std_msgs::msg::Int32::SharedPtr msg)
{
  if (!msg) {
    return;
  }
  if (msg->data == 5) {
    enable_lane_following_.store(false);
    RCLCPP_INFO(get_logger(), "Lane following disabled by sign4return=5.");
  } else if (msg->data == 6) {
    enable_lane_following_.store(true);
    RCLCPP_INFO(get_logger(), "Lane following enabled by sign4return=6.");
  }
}

int TrackDetectionNode::SetNodePara()
{
  if (!dnn_node_para_ptr_) {
    return -1;
  }
  dnn_node_para_ptr_->model_file = model_path_;
  dnn_node_para_ptr_->model_task_type =
    hobot::dnn_node::ModelTaskType::ModelInferType;
  dnn_node_para_ptr_->task_num = 1;
  dnn_node_para_ptr_->bpu_core_ids.push_back(
    bpu_core_id_ == 0 ? HB_BPU_CORE_0 : HB_BPU_CORE_1);
  return 0;
}

void TrackDetectionNode::ImageCallback(
  const hbm_img_msgs::msg::HbmMsg1080P::SharedPtr msg)
{
  if (!initialized_ || !msg || !rclcpp::ok() || !enable_lane_following_.load()) {
    return;
  }
  if (msg->width < 2 || msg->height < 2 ||
    msg->width % 2 != 0 || msg->height % 2 != 0)
  {
    RCLCPP_ERROR_THROTTLE(
      get_logger(),
      *get_clock(),
      5000,
      "NV12 image dimensions must be positive and even, got %ux%u.",
      msg->width,
      msg->height);
    return;
  }
  const size_t expected_size =
    static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height) * 3U / 2U;
  if (msg->data.size() < expected_size) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(),
      *get_clock(),
      5000,
      "NV12 buffer is too small: expected at least %zu bytes, got %zu.",
      expected_size,
      msg->data.size());
    return;
  }

  auto model = GetModel();
  if (!model) {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "Model is unavailable.");
    return;
  }

  const int source_width = static_cast<int>(msg->width);
  const int source_height = static_cast<int>(msg->height);
  int crop_top = static_cast<int>(
    std::round(source_height * kRoiTopControl / kControlHeight));
  crop_top = std::max(0, std::min(source_height - 2, EvenFloor(crop_top)));

  cv::Mat nv12(
    source_height * 3 / 2,
    source_width,
    CV_8UC1,
    const_cast<unsigned char *>(msg->data.data()));
  const cv::Range row_range(crop_top, source_height);
  const cv::Range column_range(0, source_width);
  cv::Mat model_nv12 = hobot_cv::hobotcv_crop(
    nv12,
    source_height,
    source_width,
    kModelHeight,
    kModelWidth,
    row_range,
    column_range);
  if (model_nv12.empty() ||
    model_nv12.total() < static_cast<size_t>(kModelWidth * kModelHeight * 3 / 2))
  {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 5000, "NV12 crop/resize failed.");
    return;
  }

  auto pyramid = hobot::dnn_node::ImageProc::GetNV12PyramidFromNV12Img(
    reinterpret_cast<const char *>(model_nv12.data),
    kModelHeight,
    kModelWidth,
    kModelWidth,
    kModelHeight);
  if (!pyramid) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 5000, "Failed to create NV12 pyramid input.");
    return;
  }

  auto rois = std::make_shared<std::vector<hbDNNRoi>>();
  hbDNNRoi roi;
  roi.left = 0;
  roi.top = 0;
  roi.right = kModelWidth - 1;
  roi.bottom = kModelHeight - 1;
  rois->push_back(roi);

  std::vector<std::shared_ptr<hobot::dnn_node::DNNInput>> inputs;
  for (int32_t index = 0; index < model->GetInputCount(); ++index) {
    inputs.push_back(pyramid);
  }

  auto output = std::make_shared<TrackDnnNodeOutput>();
  output->msg_header = std::make_shared<std_msgs::msg::Header>();
  output->msg_header->set__frame_id(camera_frame_id_);
  output->msg_header->set__stamp(msg->time_stamp);
  output->source_width = msg->width;
  output->source_height = msg->height;
  if (Predict(inputs, output, rois) != 0) {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "DNN Run failed.");
  }
}

int TrackDetectionNode::Predict(
  std::vector<std::shared_ptr<hobot::dnn_node::DNNInput>> & inputs,
  const std::shared_ptr<hobot::dnn_node::DnnNodeOutput> & output,
  const std::shared_ptr<std::vector<hbDNNRoi>> & rois)
{
  return Run(inputs, output, rois, true);
}

int TrackDetectionNode::PostProcess(
  const std::shared_ptr<hobot::dnn_node::DnnNodeOutput> & outputs)
{
  if (!outputs || !outputs->msg_header || outputs->output_tensors.empty()) {
    RCLCPP_ERROR(get_logger(), "DNN output is incomplete.");
    return -1;
  }

  LineCoordinateResult result;
  LineCoordinateParser parser;
  if (parser.Parse(&result, outputs->output_tensors.front()) != 0) {
    return -1;
  }

  // Publish every inference result, including low confidence. Consumers must
  // reject by confidence themselves so a stale high-confidence point is not held.
  ai_msgs::msg::PerceptionTargets message;
  message.set__header(*outputs->msg_header);
  ai_msgs::msg::Target target;
  target.set__type("track_center");
  ai_msgs::msg::Point center;
  center.set__type("midline_point");
  geometry_msgs::msg::Point32 point;
  point.set__x(result.x_roi);
  point.set__y(result.y_roi + kRoiTopControl);
  point.set__z(0.0F);
  center.point.emplace_back(point);
  center.confidence.emplace_back(result.confidence);
  target.points.emplace_back(center);
  message.targets.emplace_back(target);
  publisher_->publish(message);

  const auto track_output = std::static_pointer_cast<TrackDnnNodeOutput>(outputs);
  const double annotation_x = std::max(
    0.0,
    std::min(
      static_cast<double>(track_output->source_width - 1U),
      static_cast<double>(point.x) * track_output->source_width / kControlWidth));
  const double annotation_y = std::max(
    0.0,
    std::min(
      static_cast<double>(track_output->source_height - 1U),
      static_cast<double>(point.y) * track_output->source_height / kControlHeight));
  const bool confident =
    result.confidence >= static_cast<float>(annotation_confidence_threshold_);

  foxglove_msgs::msg::CircleAnnotation circle;
  circle.timestamp = outputs->msg_header->stamp;
  circle.position.x = annotation_x;
  circle.position.y = annotation_y;
  circle.diameter = 24.0;
  circle.thickness = 3.0;
  circle.fill_color.r = confident ? 0.0 : 1.0;
  circle.fill_color.g = confident ? 1.0 : 0.0;
  circle.fill_color.b = 0.0;
  circle.fill_color.a = 0.2;
  circle.outline_color.r = confident ? 0.0 : 1.0;
  circle.outline_color.g = confident ? 1.0 : 0.0;
  circle.outline_color.b = 0.0;
  circle.outline_color.a = 1.0;

  std::ostringstream label_stream;
  label_stream << "track conf=" << std::fixed << std::setprecision(3)
               << result.confidence;
  foxglove_msgs::msg::TextAnnotation label;
  label.timestamp = outputs->msg_header->stamp;
  label.position.x = std::min(
    static_cast<double>(track_output->source_width - 1U),
    annotation_x + 16.0);
  label.position.y = std::max(24.0, annotation_y - 12.0);
  label.text = label_stream.str();
  label.font_size = 18.0;
  label.text_color.r = 1.0;
  label.text_color.g = 1.0;
  label.text_color.b = 1.0;
  label.text_color.a = 1.0;
  label.background_color.r = 0.0;
  label.background_color.g = 0.0;
  label.background_color.b = 0.0;
  label.background_color.a = 0.65;

  foxglove_msgs::msg::ImageAnnotations annotations;
  annotations.circles.emplace_back(std::move(circle));
  annotations.texts.emplace_back(std::move(label));
  annotations_publisher_->publish(annotations);

  RCLCPP_DEBUG(
    get_logger(),
    "track_center=(%.2f, %.2f), confidence=%.4f",
    point.x,
    point.y,
    result.confidence);
  return 0;
}

int32_t LineCoordinateParser::Parse(
  LineCoordinateResult * result,
  const std::shared_ptr<hobot::dnn_node::DNNTensor> & output_tensor) const
{
  if (result == nullptr || !output_tensor) {
    RCLCPP_ERROR(rclcpp::get_logger("LineCoordinateParser"), "Invalid output tensor.");
    return -1;
  }
  auto & tensor = *output_tensor;
  int64_t element_count = 1;
  const auto & shape = tensor.properties.validShape;
  for (int index = 0; index < shape.numDimensions; ++index) {
    element_count *= shape.dimensionSize[index];
  }
  if (element_count < 3 || tensor.sysMem[0].virAddr == nullptr) {
    RCLCPP_ERROR(
      rclcpp::get_logger("LineCoordinateParser"),
      "Expected at least 3 float outputs, got %ld.",
      static_cast<long>(element_count));
    return -1;
  }

  hbSysFlushMem(&(tensor.sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  const float * values = reinterpret_cast<const float *>(tensor.sysMem[0].virAddr);
  const float x_norm = std::max(-1.0F, std::min(1.0F, values[0]));
  const float y_norm = std::max(-1.0F, std::min(1.0F, values[1]));
  result->x_roi = (x_norm + 1.0F) * 0.5F * kControlWidth;
  // Task2 uses top-left origin and y increasing downwards.
  result->y_roi = (y_norm + 1.0F) * 0.5F * kModelHeight;
  result->confidence = Sigmoid(values[2]);
  return 0;
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<TrackDetectionNode>("racing_track_detection"));
  } catch (const std::exception & error) {
    RCLCPP_FATAL(
      rclcpp::get_logger("racing_track_detection"),
      "Node startup failed: %s",
      error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}

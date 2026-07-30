#ifndef RACING_TRACK_DETECTION__RACING_TRACK_DETECTION_H_
#define RACING_TRACK_DETECTION__RACING_TRACK_DETECTION_H_

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "ai_msgs/msg/perception_targets.hpp"
#include "dnn_node/dnn_node.h"
#include "dnn_node/dnn_node_data.h"
#include "foxglove_msgs/msg/image_annotations.hpp"
#include "hbm_img_msgs/msg/hbm_msg1080_p.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp"

struct LineCoordinateResult
{
  float x_roi = 0.0F;
  float y_roi = 0.0F;
  float confidence = 0.0F;
};

struct TrackDnnNodeOutput : public hobot::dnn_node::DnnNodeOutput
{
  uint32_t source_width = 640U;
  uint32_t source_height = 480U;
};

class LineCoordinateParser
{
public:
  int32_t Parse(
    LineCoordinateResult * result,
    const std::shared_ptr<hobot::dnn_node::DNNTensor> & output_tensor) const;
};

class TrackDetectionNode : public hobot::dnn_node::DnnNode
{
public:
  explicit TrackDetectionNode(
    const std::string & node_name,
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~TrackDetectionNode() override = default;

protected:
  int SetNodePara() override;
  int PostProcess(
    const std::shared_ptr<hobot::dnn_node::DnnNodeOutput> & outputs) override;

private:
  void ImageCallback(const hbm_img_msgs::msg::HbmMsg1080P::SharedPtr msg);
  void SignCallback(const std_msgs::msg::Int32::SharedPtr msg);
  int Predict(
    std::vector<std::shared_ptr<hobot::dnn_node::DNNInput>> & inputs,
    const std::shared_ptr<hobot::dnn_node::DnnNodeOutput> & output,
    const std::shared_ptr<std::vector<hbDNNRoi>> & rois);

  std::atomic<bool> enable_lane_following_{true};
  bool initialized_{false};
  int bpu_core_id_{1};
  std::string model_path_;
  std::string sub_img_topic_{"/image_hbmem"};
  std::string output_topic_{"/racing_track_center_detection"};
  std::string sign_topic_{"/sign4return"};
  std::string camera_frame_id_{"default_usb_cam"};
  std::string annotations_topic_{"/racing_track_annotations"};
  double annotation_confidence_threshold_{0.5};

  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr sign_subscriber_;
  rclcpp::Subscription<hbm_img_msgs::msg::HbmMsg1080P>::SharedPtr image_subscriber_;
  rclcpp::Publisher<ai_msgs::msg::PerceptionTargets>::SharedPtr publisher_;
  rclcpp::Publisher<foxglove_msgs::msg::ImageAnnotations>::SharedPtr
    annotations_publisher_;
};

#endif  // RACING_TRACK_DETECTION__RACING_TRACK_DETECTION_H_

import os, json, socket, threading, time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry

WIN_IP = os.environ.get("WIN_IP", "")        # 예: awk '/nameserver/{print $2; exit}' /etc/resolv.conf
PORT_CMD = int(os.environ.get("WIN_PORT_CMD", "50051"))
PORT_WRITER = int(os.environ.get("WIN_PORT_WRITER", "50052"))

class Bridge(Node):
    def __init__(self):
        super().__init__("go2_bridge")
        self.pub_pose = self.create_publisher(PoseStamped, "/unitree_go2/pose", 10)
        self.pub_odom = self.create_publisher(Odometry, "/unitree_go2/odom", 10)
        self.sub_cmd = self.create_subscription(Twist, "/unitree_go2/cmd_vel", self.on_cmd, 10)
        # 송신 소켓(cmd_vel)
        self.sock_cmd = socket.create_connection((WIN_IP, PORT_CMD))
        self.out = self.sock_cmd.makefile("w")
        self.get_logger().info(f"cmd_vel connected to {WIN_IP}:{PORT_CMD}")
        # 수신 소켓(odom/pose)용: writer tap에 연결
        self.sock_writer = socket.create_connection((WIN_IP, PORT_WRITER))
        self.inp = self.sock_writer.makefile("r")
        self.get_logger().info(f"writer tap connected to {WIN_IP}:{PORT_WRITER}")
        threading.Thread(target=self.reader_loop, daemon=True).start()

    def on_cmd(self, msg: Twist):
        payload = {"type": "cmd_vel", "vx": msg.linear.x, "vy": msg.linear.y, "wz": msg.angular.z}
        try:
            self.out.write(json.dumps(payload) + "\n")
            self.out.flush()
        except Exception as e:
            self.get_logger().error(f"cmd tx fail: {e}")

    def reader_loop(self):
        while rclpy.ok():
            line = self.inp.readline()
            if not line:
                time.sleep(0.05)
                continue
            try:
                msg = json.loads(line.strip())
                if msg.get("type") == "odom_pose":
                    self.pub_from_odom_pose(msg)
            except Exception as e:
                self.get_logger().warn(f"rx parse fail: {e}")

    def pub_from_odom_pose(self, m):
        # Pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = m.get("frame_id", "map")
        pos = m["pos"]; q = m["quat_wxyz"]
        pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = pos
        # ROS는 x,y,z,w 순. 원본은 w,x,y,z이므로 변환
        pose_msg.pose.orientation.x = q[1]; pose_msg.pose.orientation.y = q[2]
        pose_msg.pose.orientation.z = q[3]; pose_msg.pose.orientation.w = q[0]
        self.pub_pose.publish(pose_msg)
        # Odom
        odom = Odometry()
        odom.header.stamp = pose_msg.header.stamp
        odom.header.frame_id = m.get("frame_id", "map")
        odom.child_frame_id = m.get("child_frame_id", "unitree_go2/base_link")
        odom.pose.pose = pose_msg.pose
        lin_b = m["lin_vel_b"]; ang_b = m["ang_vel_b"]
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = lin_b
        odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z = ang_b
        self.pub_odom.publish(odom)

def main():
    rclpy.init()
    node = Bridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
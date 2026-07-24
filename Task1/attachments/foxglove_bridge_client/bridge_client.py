#!/usr/bin/env python3
"""OriginCar keyboard teleoperation through Foxglove Bridge.

The key-state behavior follows references/foxglove/src/bridge_client.py. The
transport is the Foxglove WebSocket protocol instead of rosbridge/roslibpy.
No ROS 2 installation is required on the control computer.
"""

from __future__ import annotations

import argparse
import json
import signal
import struct
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set

try:
    import yaml
except ImportError:
    yaml = None

try:
    import websocket
except ImportError:
    websocket = None


FOXGLOVE_SUBPROTOCOLS = ("foxglove.sdk.v1", "foxglove.websocket.v1")
CLIENT_MESSAGE_DATA_OPCODE = 1
SERVER_MESSAGE_DATA_OPCODE = 1

CMD_VEL_CHANNEL_ID = 1
MAP_CHANNEL_ID = 2
SIGN_CHANNEL_ID = 3
SIGN_SUBSCRIPTION_ID = 1

ZERO_TWIST = {
    "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
    "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
}


def application_dir() -> Path:
    """Return the source folder or the folder containing the frozen executable."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


@contextmanager
def terminal_echo_disabled():
    """Hide held-key echo on POSIX terminals and always restore terminal state."""
    if sys.platform == "win32" or not sys.stdin.isatty():
        yield
        return

    try:
        import termios
    except ImportError:
        yield
        return

    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    modified = termios.tcgetattr(fd)
    modified[3] &= ~termios.ECHO

    termios.tcsetattr(fd, termios.TCSANOW, modified)
    try:
        yield
    finally:
        try:
            termios.tcflush(fd, termios.TCIFLUSH)
        finally:
            termios.tcsetattr(fd, termios.TCSANOW, original)


def pause_before_frozen_exit(disabled: bool) -> None:
    """Keep a double-clicked console open so the final status remains visible."""
    if disabled or not getattr(sys, "frozen", False) or not sys.stdin.isatty():
        return
    try:
        input("\n按回车关闭窗口...")
    except (EOFError, KeyboardInterrupt):
        pass


def twist(linear_x: float = 0.0, angular_z: float = 0.0) -> Dict[str, Any]:
    return {
        "linear": {"x": linear_x, "y": 0.0, "z": 0.0},
        "angular": {"x": 0.0, "y": 0.0, "z": angular_z},
    }


def resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def load_config(config_path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("缺少 PyYAML，请先执行 pip install -r requirements.txt")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"配置文件顶层必须是对象: {config_path}")
    return config


class FoxgloveBridgeConnection:
    """Minimal Foxglove WebSocket client with JSON client publishing."""

    def __init__(
        self,
        url: str,
        sign_topic: Optional[str],
        on_sign: Callable[[int], None],
        connect_timeout: float = 5.0,
    ) -> None:
        if websocket is None:
            raise RuntimeError(
                "缺少 websocket-client，请先执行 pip install -r requirements.txt"
            )
        self.url = url
        self.sign_topic = sign_topic
        self.on_sign = on_sign
        self.connect_timeout = connect_timeout
        self.socket: Any = None
        self.send_lock = threading.Lock()
        self.server_info_event = threading.Event()
        self.closed_event = threading.Event()
        self.receiver_thread: Optional[threading.Thread] = None
        self.server_info: Dict[str, Any] = {}
        self.sign_channel_id: Optional[int] = None
        self.sign_subscribed = False
        self.advertised_channel_ids: list[int] = []
        self.receiver_error: Optional[BaseException] = None

    def connect(self) -> None:
        self.socket = websocket.create_connection(
            self.url,
            subprotocols=list(FOXGLOVE_SUBPROTOCOLS),
            timeout=self.connect_timeout,
        )
        if self.socket.getsubprotocol() not in FOXGLOVE_SUBPROTOCOLS:
            self.socket.close()
            raise RuntimeError(
                "服务端未协商 Foxglove SDK/WebSocket 子协议；"
                "请确认连接的是 Foxglove Bridge"
            )
        self.socket.settimeout(0.5)
        self.receiver_thread = threading.Thread(
            target=self._receive_loop,
            name="foxglove-receiver",
            daemon=True,
        )
        self.receiver_thread.start()
        if not self.server_info_event.wait(self.connect_timeout):
            self.close(send_stop=False)
            raise TimeoutError("连接成功，但未收到 Foxglove serverInfo")

        capabilities = self.server_info.get("capabilities", [])
        encodings = self.server_info.get("supportedEncodings", [])
        if "clientPublish" not in capabilities:
            self.close(send_stop=False)
            raise RuntimeError("Foxglove Bridge 未启用 clientPublish capability")
        if encodings and "json" not in encodings:
            self.close(send_stop=False)
            raise RuntimeError("Foxglove Bridge 不支持 JSON client publishing")

    def advertise(self, channels: list[Dict[str, Any]]) -> None:
        self._send_text({"op": "advertise", "channels": channels})
        self.advertised_channel_ids = [int(channel["id"]) for channel in channels]

    def unadvertise(self, channel_ids: list[int]) -> None:
        if self.socket is not None and not self.closed_event.is_set():
            self._send_text({"op": "unadvertise", "channelIds": channel_ids})

    def publish_json(self, channel_id: int, message: Dict[str, Any]) -> None:
        data = json.dumps(
            message, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        frame = struct.pack("<BI", CLIENT_MESSAGE_DATA_OPCODE, channel_id) + data
        self._send_binary(frame)

    def close(self, send_stop: bool = True) -> None:
        if self.socket is None or self.closed_event.is_set():
            return
        if send_stop:
            try:
                self.publish_json(CMD_VEL_CHANNEL_ID, ZERO_TWIST)
            except Exception:
                pass
        try:
            self.unadvertise(self.advertised_channel_ids)
        except Exception:
            pass
        self.closed_event.set()
        try:
            self.socket.close()
        except Exception:
            pass

    def _send_text(self, message: Dict[str, Any]) -> None:
        payload = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
        with self.send_lock:
            if self.socket is None or self.closed_event.is_set():
                raise ConnectionError("Foxglove Bridge 连接已关闭")
            self.socket.send(payload)

    def _send_binary(self, payload: bytes) -> None:
        with self.send_lock:
            if self.socket is None or self.closed_event.is_set():
                raise ConnectionError("Foxglove Bridge 连接已关闭")
            self.socket.send(payload, opcode=websocket.ABNF.OPCODE_BINARY)

    def _receive_loop(self) -> None:
        try:
            while not self.closed_event.is_set():
                try:
                    message = self.socket.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                if message in (None, ""):
                    continue
                if isinstance(message, str):
                    self._handle_text(message)
                elif isinstance(message, (bytes, bytearray)):
                    self._handle_binary(bytes(message))
        except websocket.WebSocketConnectionClosedException:
            pass
        except BaseException as error:
            self.receiver_error = error
            if not self.closed_event.is_set():
                print(f"\nFoxglove 接收线程异常: {error}", file=sys.stderr)
        finally:
            self.closed_event.set()
            self.server_info_event.set()

    def _handle_text(self, payload: str) -> None:
        message = json.loads(payload)
        operation = message.get("op")
        if operation == "serverInfo":
            self.server_info = message
            self.server_info_event.set()
        elif operation == "advertise":
            self._handle_advertise(message.get("channels", []))
        elif operation == "unadvertise":
            removed = set(message.get("channelIds", []))
            if self.sign_channel_id in removed:
                self.sign_channel_id = None
                self.sign_subscribed = False
        elif operation == "status":
            level = int(message.get("level", 0))
            output = sys.stderr if level >= 1 else sys.stdout
            print(f"Foxglove status[{level}]: {message.get('message', '')}", file=output)

    def _handle_advertise(self, channels: list[Dict[str, Any]]) -> None:
        if not self.sign_topic or self.sign_subscribed:
            return
        for channel in channels:
            if channel.get("topic") == self.sign_topic:
                self.sign_channel_id = int(channel["id"])
                self._send_text(
                    {
                        "op": "subscribe",
                        "subscriptions": [
                            {
                                "id": SIGN_SUBSCRIPTION_ID,
                                "channelId": self.sign_channel_id,
                            }
                        ],
                    }
                )
                self.sign_subscribed = True
                print(f"已订阅遥测信号 {self.sign_topic}")
                return

    def _handle_binary(self, payload: bytes) -> None:
        if len(payload) < 13 or payload[0] != SERVER_MESSAGE_DATA_OPCODE:
            return
        subscription_id = struct.unpack_from("<I", payload, 1)[0]
        if subscription_id != SIGN_SUBSCRIPTION_ID:
            return
        cdr = payload[13:]
        if len(cdr) < 8:
            return
        encapsulation = cdr[:2]
        endian = "<" if encapsulation in (b"\x00\x01", b"\x00\x03") else ">"
        value = struct.unpack_from(f"{endian}i", cdr, 4)[0]
        self.on_sign(value)


class OriginCarTeleop:
    """Keyboard state machine matching the original bridge_client behavior."""

    def __init__(
        self,
        connection: FoxgloveBridgeConnection,
        config: Dict[str, Any],
        config_path: Path,
    ) -> None:
        self.connection = connection
        self.config = config
        self.config_path = config_path
        self.linear = float(config.get("linear", 0.3))
        self.angular = float(config.get("angular", 0.8))
        self.delta = float(config.get("delta", 0.1))
        self.publish_rate_hz = float(config.get("publish_rate_hz", 20.0))
        self.cmd_vel_topic = str(config.get("cmd_vel_topic", "/cmd_vel"))
        self.map_topic = str(config.get("map_topic", "/map"))
        self.sign_publish_topic = str(
            config.get("sign_publish_topic", "/sign_foxglove")
        )
        self.map_enabled = bool(config.get("map_enabled", False))
        self.signals_enabled = bool(config.get("signals_enabled", False))
        self.map_path = resolve_path(str(config.get("map_path", "map.png")), config_path)
        self.map_width_m = float(config.get("map_width_m", 5.0))
        self.map_message: Optional[Dict[str, Any]] = None
        self.pressed: Set[str] = set()
        self.state_lock = threading.Lock()
        self.active = False
        self.stop_event = threading.Event()
        self.listener: Any = None

    def advertise(self) -> None:
        channels = [
            {
                "id": CMD_VEL_CHANNEL_ID,
                "topic": self.cmd_vel_topic,
                "encoding": "json",
                "schemaName": "geometry_msgs/msg/Twist",
            }
        ]
        if self.map_enabled:
            channels.append(
                {
                    "id": MAP_CHANNEL_ID,
                    "topic": self.map_topic,
                    "encoding": "json",
                    "schemaName": "nav_msgs/msg/OccupancyGrid",
                }
            )
        if self.signals_enabled:
            channels.append(
                {
                    "id": SIGN_CHANNEL_ID,
                    "topic": self.sign_publish_topic,
                    "encoding": "json",
                    "schemaName": "std_msgs/msg/Int32",
                }
            )
        self.connection.advertise(channels)
        time.sleep(0.2)
        self.publish_stop()

        if self.map_enabled:
            self.map_message = self._load_map()
            self.publish_map()
        if self.signals_enabled:
            self.connection.publish_json(SIGN_CHANNEL_ID, {"data": 0})

    def run(self) -> None:
        try:
            from pynput import keyboard
        except ImportError as error:
            raise RuntimeError(
                "缺少 pynput，请先执行 pip install -r requirements.txt"
            ) from error

        self._keyboard_module = keyboard
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self.listener.start()
        self.show_help()
        print("\n按 [R] 激活遥操；当前处于暂停状态。")

        period = 1.0 / self.publish_rate_hz
        next_publish = time.monotonic()
        while not self.stop_event.is_set():
            if self.connection.closed_event.is_set():
                raise ConnectionError("Foxglove Bridge 连接已断开")
            if self.active:
                self.connection.publish_json(
                    CMD_VEL_CHANNEL_ID, self._current_twist()
                )
            next_publish += period
            delay = next_publish - time.monotonic()
            if delay > 0:
                self.stop_event.wait(delay)
            else:
                next_publish = time.monotonic()

    def shutdown(self) -> None:
        self.stop_event.set()
        self.active = False
        try:
            self.publish_stop()
        except Exception:
            pass
        if self.listener is not None:
            self.listener.stop()

    def publish_stop(self) -> None:
        self.connection.publish_json(CMD_VEL_CHANNEL_ID, ZERO_TWIST)

    def publish_map(self) -> None:
        if self.map_message is None:
            return
        now_ns = time.time_ns()
        self.map_message["header"]["stamp"] = {
            "sec": now_ns // 1_000_000_000,
            "nanosec": now_ns % 1_000_000_000,
        }
        self.connection.publish_json(MAP_CHANNEL_ID, self.map_message)
        print(f"已发布地图 {self.map_topic}")

    def handle_sign(self, value: int) -> None:
        if value == 5:
            self.active = True
            print("\n接收到[正在C区进行遥测]信号，已激活键盘遥操")
        elif value == 6:
            self.active = False
            self.publish_stop()
            print("\n接收到[C区出口结束遥测]信号，已暂停键盘遥操")
        if self.signals_enabled:
            self.connection.publish_json(SIGN_CHANNEL_ID, {"data": value})

    def _current_twist(self) -> Dict[str, Any]:
        with self.state_lock:
            keys = set(self.pressed)

        if "w" in keys:
            if "a" in keys:
                return twist(self.linear, self.angular)
            if "d" in keys:
                return twist(self.linear, -self.angular)
            return twist(self.linear, 0.0)
        if "s" in keys:
            if "d" in keys:
                return twist(-self.linear, self.angular)
            if "a" in keys:
                return twist(-self.linear, -self.angular)
            return twist(-self.linear, 0.0)
        if "a" in keys:
            return twist(0.0, self.angular)
        if "d" in keys:
            return twist(0.0, -self.angular)
        return ZERO_TWIST

    def _on_press(self, key: Any) -> Optional[bool]:
        name = self._key_name(key)
        if name is None:
            return None
        with self.state_lock:
            first_press = name not in self.pressed
            self.pressed.add(name)
            motion_held = bool(self.pressed.intersection({"w", "a", "s", "d"}))
        if not first_press:
            return None

        if name == "esc":
            self.shutdown()
            return False
        if name == "r":
            self.active = True
            print("\n已激活键盘遥操")
        elif name == "p":
            self.active = False
            self.publish_stop()
            print("\n已暂停键盘遥操；按 [R] 继续")
        elif name == "t":
            self.show_help()
        elif name == "m" and self.map_enabled:
            self.publish_map()
        elif not motion_held:
            if name == "up":
                self.linear += self.delta
                print(f"线速度设置为: {self.linear:.2f} m/s")
            elif name == "down":
                if self.linear - self.delta < 0:
                    print(f"速度设置失败（当前线速度: {self.linear:.2f} m/s）")
                else:
                    self.linear -= self.delta
                    print(f"线速度设置为: {self.linear:.2f} m/s")
            elif name == "left":
                if self.angular - self.delta < 0:
                    print(f"转角设置失败（当前角速度值: {self.angular:.2f}）")
                else:
                    self.angular -= self.delta
                    print(f"转向值设置为: {self.angular:.2f}")
            elif name == "right":
                self.angular += self.delta
                print(f"转向值设置为: {self.angular:.2f}")
        return None

    def _on_release(self, key: Any) -> None:
        name = self._key_name(key)
        if name is None:
            return
        with self.state_lock:
            self.pressed.discard(name)

    def _key_name(self, key: Any) -> Optional[str]:
        keyboard = self._keyboard_module
        special = {
            keyboard.Key.up: "up",
            keyboard.Key.down: "down",
            keyboard.Key.left: "left",
            keyboard.Key.right: "right",
            keyboard.Key.esc: "esc",
        }
        if key in special:
            return special[key]
        char = getattr(key, "char", None)
        return char.lower() if isinstance(char, str) else None

    def _load_map(self) -> Dict[str, Any]:
        try:
            from PIL import Image, ImageOps
        except ImportError as error:
            raise RuntimeError(
                "启用地图发布时需要 Pillow，请先执行 pip install -r requirements.txt"
            ) from error
        if not self.map_path.is_file():
            raise FileNotFoundError(f"地图文件不存在: {self.map_path}")
        image = Image.open(self.map_path).convert("L")
        image = ImageOps.flip(image)
        width, height = image.size
        occupancy = [0 if pixel < 128 else -1 for pixel in image.getdata()]
        return {
            "header": {
                "stamp": {"sec": 0, "nanosec": 0},
                "frame_id": "odom_combined",
            },
            "info": {
                "map_load_time": {"sec": 0, "nanosec": 0},
                "resolution": self.map_width_m / width,
                "width": width,
                "height": height,
                "origin": {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            },
            "data": occupancy,
        }

    def show_help(self) -> None:
        print(
            f"""
控制：
  R       激活遥操
  P       暂停遥操并发送停止
  W / S   前进 / 后退，当前线速度 {self.linear:.2f} m/s
  W+A/D   前进左转 / 前进右转
  S+D/A   后退左转 / 后退右转（保持原 bridge_client 映射）
  A / D   只发送转向值；Ackermann 小车静止时舵机不会动作
  ↑ / ↓   增加 / 减小线速度
  ← / →   减小 / 增加转向值
  T       显示帮助
  M       重新发布地图（仅 map_enabled=true）
  Esc     发送停止并退出
"""
        )


def print_mapping(config: Dict[str, Any]) -> None:
    linear = float(config.get("linear", 0.3))
    angular = float(config.get("angular", 0.8))
    rows = {
        "W": twist(linear, 0.0),
        "S": twist(-linear, 0.0),
        "A": twist(0.0, angular),
        "D": twist(0.0, -angular),
        "W+A": twist(linear, angular),
        "W+D": twist(linear, -angular),
        "S+D": twist(-linear, angular),
        "S+A": twist(-linear, -angular),
        "无按键": ZERO_TWIST,
    }
    for keys, message in rows.items():
        print(
            f"{keys:6s} linear.x={message['linear']['x']: .3f} "
            f"angular.z={message['angular']['z']: .3f}"
        )


def parse_args() -> argparse.Namespace:
    default_config = application_dir() / "config.yaml"
    parser = argparse.ArgumentParser(
        description="通过 Foxglove Bridge 控制 OriginCar"
    )
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--url", help="覆盖配置中的 bridge_url")
    parser.add_argument(
        "--check",
        action="store_true",
        help="连接、advertise 并只发送一次零速度，不监听键盘",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="不连接小车，只打印按键映射",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="打包程序退出时不等待回车，适合从已有终端或脚本启动",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    if args.url:
        config["bridge_url"] = args.url
    if args.dry_run:
        print_mapping(config)
        return 0

    bridge_url = str(config.get("bridge_url", "ws://127.0.0.1:8765"))
    signals_enabled = bool(config.get("signals_enabled", False))
    sign_subscribe_topic = (
        str(config.get("sign_subscribe_topic", "/sign4return"))
        if signals_enabled
        else None
    )

    teleop_holder: Dict[str, OriginCarTeleop] = {}

    def on_sign(value: int) -> None:
        teleop = teleop_holder.get("teleop")
        if teleop is not None:
            teleop.handle_sign(value)

    connection = FoxgloveBridgeConnection(
        bridge_url,
        sign_topic=sign_subscribe_topic,
        on_sign=on_sign,
    )
    teleop = OriginCarTeleop(connection, config, config_path)
    teleop_holder["teleop"] = teleop

    def request_shutdown(_signum: int, _frame: Any) -> None:
        teleop.shutdown()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    try:
        print(f"正在连接 {bridge_url}")
        connection.connect()
        server_name = connection.server_info.get("name") or "Foxglove Bridge"
        print(
            f"已连接 {server_name}，"
            "clientPublish/json 可用"
        )
        teleop.advertise()
        if args.check:
            print("安全检查通过：已 advertise /cmd_vel，并且只发送了零速度")
            return 0
        with terminal_echo_disabled():
            teleop.run()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1
    finally:
        teleop.shutdown()
        connection.close(send_stop=True)


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    finally:
        pause_before_frozen_exit("--no-pause" in sys.argv)
    raise SystemExit(exit_code)

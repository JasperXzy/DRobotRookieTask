# Foxglove Bridge keyboard client

这是 `references/foxglove/src/bridge_client.py` 的 Foxglove Bridge 版本。它不再
连接 rosbridge `:9090`，而是直接连接 Foxglove Bridge `:8765`，通过
`clientPublish + json` 发布 `geometry_msgs/msg/Twist`。

客户端优先使用 Foxglove Bridge 3.x 的 `foxglove.sdk.v1` 子协议，并兼容旧版
`foxglove.websocket.v1`。

## 安装

建议使用 Python 虚拟环境：

```bash
cd Task1/attachments/foxglove_bridge_client
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

macOS 首次运行时，需要在“系统设置 → 隐私与安全性 → 辅助功能”中允许当前
Terminal/Codex 读取键盘。否则 `pynput` 无法收到全局按键。

## 运行

先启动 rdk-x5 上的底盘和 Foxglove Bridge，再在控制电脑运行：

```bash
python3 bridge_client.py
```

正式遥操前请导入 `../foxglove_task1.json`，或删除当前布局中的内置 Teleop。
不要让内置 Teleop 与本客户端同时 advertise `/cmd_vel`。

也可以覆盖连接地址：

```bash
python3 bridge_client.py --url ws://<小车IP>:8765
```

离线检查映射：

```bash
python3 bridge_client.py --dry-run
```

## 构建

Windows 和 macOS 必须在各自平台上构建。构建配置保存在 `packaging/`，
PyInstaller 生成的 `build/` 和 `dist/` 只作为本地临时目录。

macOS：

```bash
python3 -m venv .venv-build
source .venv-build/bin/activate
python -m pip install --upgrade pip pyinstaller
python -m pip install -r requirements.txt
python -m PyInstaller --noconfirm --clean \
  packaging/DRobotTeleop.macos.spec

mkdir -p ../../../release/DRobotTeleop/macos-arm64
cp dist/DRobotTeleop ../../../release/DRobotTeleop/macos-arm64/
cp config.yaml map.png ../../../release/DRobotTeleop/macos-arm64/
```

Windows PowerShell：

```powershell
py -3 -m venv .venv-build
.\.venv-build\Scripts\python.exe -m pip install --upgrade pip pyinstaller
.\.venv-build\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-build\Scripts\python.exe -m PyInstaller --noconfirm --clean `
  packaging\DRobotTeleop.windows.spec

New-Item -ItemType Directory -Force `
  ..\..\..\release\DRobotTeleop\windows-x86_64
Copy-Item dist\DRobotTeleop.exe `
  ..\..\..\release\DRobotTeleop\windows-x86_64\
Copy-Item config.yaml,map.png `
  ..\..\..\release\DRobotTeleop\windows-x86_64\
```

## 双击运行打包程序

PyInstaller 打包后，不要把二进制保留在源码目录的 `dist/` 中。将可执行
文件、`config.yaml` 和 `map.png` 按平台放到仓库根目录的 `release/`：

```text
release/DRobotTeleop/
├── macos-arm64/
│   ├── DRobotTeleop
│   ├── config.yaml
│   └── map.png
└── windows-x86_64/
    ├── DRobotTeleop.exe
    ├── config.yaml
    └── map.png
```

打包状态下，客户端会使用可执行文件所在目录作为附件目录，因此可以直接双击
`DRobotTeleop` 或 `DRobotTeleop.exe`，不再需要手动传入 `--config`。

打包程序退出前默认显示“按回车关闭窗口”，避免双击启动后错误信息一闪而过。
从已有终端或自动化脚本启动时，可以跳过等待：

```bash
./DRobotTeleop --no-pause
```

macOS/Linux 运行遥操时会暂时关闭当前终端的键盘回显，所以持续按住 WASD
不会在终端中产生连续字母；退出时会自动恢复终端设置。

## 按键

- `R`：激活键盘控制。
- `P`：暂停并立即发送零速度。
- `W` / `S`：前进 / 后退。
- `W+A` / `W+D`：前进左转 / 前进右转。
- `S+D` / `S+A`：后退左转 / 后退右转，保持原脚本的映射。
- `A` / `D`：发布纯转向命令；由于 STM32 Ackermann 换算要求
  `Vx != 0 && Vz != 0`，车辆静止时舵机不动作，这是预期行为。
- `↑` / `↓`：增加 / 减小线速度。
- `←` / `→`：减小 / 增加转向值。
- `T`：显示帮助。
- `M`：重新发布地图，仅当 `map_enabled: true`。
- `Esc`：发送零速度并退出。

客户端激活后以 20 Hz 持续发布；没有按下 WASD 时持续发布零速度。当前板端
保持原始 `origincar_base`，没有失联看门狗，因此网络突然中断或进程被强杀时
不能保证自动停车；实车测试必须保留人工急停。

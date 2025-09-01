from isaacsim import SimulationApp
import os, math, time, json, threading, socketserver
import hydra

# (중요) Windows 시뮬은 ROS2 미사용
USE_SOCKET_BRIDGE = os.environ.get("USE_SOCKET_BRIDGE", "1") not in ("0", "false", "False")
FILE_PATH = os.path.join(os.path.dirname(__file__), "cfg")

# 간단한 TCP 핸들러: cmd_vel 수신
class _CmdHandler(socketserver.StreamRequestHandler):
    def handle(self):
        import go2.go2_ctrl as go2_ctrl
        for line in self.rfile:
            try:
                msg = json.loads(line.decode("utf-8").strip())
                if msg.get("type") == "cmd_vel":
                    vx = float(msg.get("vx", 0.0))
                    vy = float(msg.get("vy", 0.0))
                    wz = float(msg.get("wz", 0.0))
                    # env 0 기준, 필요 시 다중 env 확장
                    go2_ctrl.base_vel_cmd_input[0][0] = vx
                    go2_ctrl.base_vel_cmd_input[0][1] = vy
                    go2_ctrl.base_vel_cmd_input[0][2] = wz
            except Exception as e:
                print(f"[TCP] bad message: {e}")

def start_cmd_server(host="127.0.0.1", port=50051):
    server = socketserver.ThreadingTCPServer((host, port), _CmdHandler)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    print(f"[TCP] listening on {host}:{port}")
    return server

@hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)
def run_simulator(cfg):
    # Omniverse 앱 시작
    simulation_app = SimulationApp({"headless": False,
                                    "anti_aliasing": cfg.sim_app.anti_aliasing,
                                    "width": cfg.sim_app.width, "height": cfg.sim_app.height,
                                    "hide_ui": cfg.sim_app.hide_ui})
    import omni, carb, torch
    import go2.go2_ctrl as go2_ctrl
    from go2.go2_env import Go2RSLEnvCfg, camera_follow
    import env.sim_env as sim_env
    import go2.go2_sensors as go2_sensors

    # 환경/정책
    go2_env_cfg = Go2RSLEnvCfg()
    go2_env_cfg.scene.num_envs = cfg.num_envs
    go2_env_cfg.decimation = math.ceil(1.0 / go2_env_cfg.sim.dt / cfg.freq)
    go2_env_cfg.sim.render_interval = go2_env_cfg.decimation
    go2_ctrl.init_base_vel_cmd(cfg.num_envs)
    env, policy = go2_ctrl.get_rsl_rough_policy(go2_env_cfg)

    # 환경 로딩
    if cfg.env_name == "obstacle-dense":
        sim_env.create_obstacle_dense_env()
    elif cfg.env_name == "obstacle-medium":
        sim_env.create_obstacle_medium_env()
    elif cfg.env_name == "obstacle-sparse":
        sim_env.create_obstacle_sparse_env()
    elif cfg.env_name == "warehouse":
        sim_env.create_warehouse_env()
    elif cfg.env_name == "warehouse-forklifts":
        sim_env.create_warehouse_forklifts_env()
    elif cfg.env_name == "warehouse-shelves":
        sim_env.create_warehouse_shelves_env()
    elif cfg.env_name == "full-warehouse":
        sim_env.create_full_warehouse_env()

    # 센서(필요 시)
    sm = go2_sensors.SensorManager(cfg.num_envs)
    lidar_annotators = sm.add_rtx_lidar()
    cameras = sm.add_camera(cfg.freq)

    # 키보드
    system_input = carb.input.acquire_input_interface()
    system_input.subscribe_to_keyboard_events(
        omni.appwindow.get_default_app_window().get_keyboard(), go2_ctrl.sub_keyboard_event)

    # TCP 서버 시작
    server = None
    client_writer = {"fp": None}  # 간단 공유 객체
    if USE_SOCKET_BRIDGE:
        server = start_cmd_server(host="127.0.0.1", port=50051)
        # 간단: 최초 연결 시까지 기다리지 않고, 송신 실패를 try/except로 무시

    # 실행
    sim_step_dt = float(go2_env_cfg.sim.dt * go2_env_cfg.decimation)
    obs, _ = env.reset()
    last_pub = 0.0
    pub_hz = 30.0  # odom/pose 송신 주기

    def send_json(msg):
        # 간단 송신: 현재는 연결 추적 없이 실패 시 무시(브리지 먼저 접속 권장)
        try:
            if not client_writer["fp"]:
                return
            client_writer["fp"].write(json.dumps(msg) + "\n")
            client_writer["fp"].flush()
        except Exception:
            client_writer["fp"] = None  # 끊기면 다음에 재설정

    # 클라이언트 writer를 얻기 위한 보조 서버(선택): 첫 접속자에게만 writer 저장
    class _WriterTap(socketserver.StreamRequestHandler):
        def handle(self):
            client_writer["fp"] = self.wfile
            while self.rfile.readline():
                pass

    writer_server = None
    if USE_SOCKET_BRIDGE:
        writer_server = socketserver.ThreadingTCPServer(("127.0.0.1", 50052), _WriterTap)
        threading.Thread(target=writer_server.serve_forever, daemon=True).start()
        print("[TCP] writer tap on 127.0.0.1:50052")

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            # Camera follow
            if cfg.camera_follow:
                camera_follow(env)

            # 주기적으로 pose/odom 송신
            now = time.time()
            if USE_SOCKET_BRIDGE and (now - last_pub) >= (1.0 / pub_hz):
                last_pub = now
                data = env.unwrapped.scene["unitree_go2"].data
                # env 0 기준
                pos = data.root_state_w[0, :3].cpu().numpy().tolist()
                quat_wxyz = data.root_state_w[0, 3:7].cpu().numpy().tolist()
                lin_b = data.root_lin_vel_b[0].cpu().numpy().tolist()
                ang_b = data.root_ang_vel_b[0].cpu().numpy().tolist()
                send_json({
                    "type": "odom_pose",
                    "pos": pos,
                    "quat_wxyz": quat_wxyz,
                    "lin_vel_b": lin_b,
                    "ang_vel_b": ang_b,
                    "frame_id": "map",
                    "child_frame_id": "unitree_go2/base_link"
                })

            # 루프 타임 제어
            elapsed = time.time() - start_time
            if elapsed < sim_step_dt:
                time.sleep(sim_step_dt - elapsed)

    if writer_server:
        writer_server.shutdown()
    if server:
        server.shutdown()
    simulation_app.close()

if __name__ == "__main__":
    run_simulator()
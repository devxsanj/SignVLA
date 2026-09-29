"""
RoArm-M2 Pro Hardware Controller (Waveshare Serial Protocol)
Communicates via USB-UART using JSON line commands.
"""
import sys
import time
import json
from pathlib import Path
import serial
import serial.tools.list_ports

# Auto re-exec inside virtual environment if invoked with global system Python
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VENV_PY = _PROJECT_ROOT / ".venv" / "bin" / "python"
if not _VENV_PY.exists():
    _VENV_PY = _PROJECT_ROOT / "venv" / "bin" / "python"

if _VENV_PY.exists() and sys.prefix == sys.base_prefix:
    import os
    os.execv(str(_VENV_PY), [str(_VENV_PY)] + sys.argv)


def find_roarm_port():
    """Auto-detects USB serial port connected to ESP32 / RoArm on macOS."""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        desc = (port.description or "").lower()
        dev = port.device.lower()
        if "usb" in dev or "ch340" in desc or "cp210" in desc or "uart" in desc:
            return port.device
    if ports:
        return ports[0].device
    return None


class RoArmM2Hardware:
    def __init__(self, port=None, baudrate=115200, timeout=1.0):
        self.port = port or find_roarm_port()
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None

    def connect(self):
        if not self.port:
            print("[ROARM-HW] No serial port detected! Please plug in RoArm-M2 Pro via USB.")
            return False
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2.0)  # Wait for ESP32 reboot / initialize
            print(f"[ROARM-HW] Connected to RoArm-M2 Pro on {self.port} at {self.baudrate} baud.")
            return True
        except Exception as e:
            print(f"[ROARM-HW] Connection error on {self.port}: {e}")
            return False

    def send_cmd(self, cmd_dict):
        """Sends a JSON command to RoArm-M2 Pro ESP32."""
        if not self.ser or not self.ser.is_open:
            return None
        line = json.dumps(cmd_dict) + "\n"
        self.ser.write(line.encode("utf-8"))
        self.ser.flush()
        # Read reply if available
        try:
            reply = self.ser.readline().decode("utf-8", errors="ignore").strip()
            return reply
        except Exception:
            return None

    def set_joints(self, b_rad, s_rad, e_rad, w_rad, spd=0):
        """Sets all 4 arm joint angles (in radians)."""
        # Waveshare JSON command for joint control: T=101
        self.send_cmd({"T": 101, "joint": 1, "rad": round(b_rad, 3), "spd": spd})
        self.send_cmd({"T": 101, "joint": 2, "rad": round(s_rad, 3), "spd": spd})
        self.send_cmd({"T": 101, "joint": 3, "rad": round(e_rad, 3), "spd": spd})
        self.send_cmd({"T": 101, "joint": 4, "rad": round(w_rad, 3), "spd": spd})

    def set_gripper(self, open_ratio):
        """open_ratio: 0.0 (closed) to 1.0 (fully open)"""
        # Waveshare gripper command: T=103 or joint 5
        rad = open_ratio * 1.57  # ~90 degrees range
        self.send_cmd({"T": 103, "joint": 1, "rad": round(rad, 3)})

    def move_xyz(self, x, y, z, t=0):
        """Cartesian coordinates command in mm."""
        self.send_cmd({"T": 102, "x": round(x, 1), "y": round(y, 1), "z": round(z, 1), "t": t})

    def home(self):
        print("[ROARM-HW] Moving to Home Pose")
        self.send_cmd({"T": 100})
        time.sleep(1.0)

    def wave(self):
        print("[ROARM-HW] Waving")
        for _ in range(2):
            self.send_cmd({"T": 101, "joint": 1, "rad": 0.35, "spd": 200})
            time.sleep(0.4)
            self.send_cmd({"T": 101, "joint": 1, "rad": -0.35, "spd": 200})
            time.sleep(0.4)
        self.home()

    def nod(self):
        print("[ROARM-HW] Nodding (Yes)")
        for _ in range(2):
            self.send_cmd({"T": 101, "joint": 4, "rad": 0.5, "spd": 150})
            time.sleep(0.3)
            self.send_cmd({"T": 101, "joint": 4, "rad": -0.2, "spd": 150})
            time.sleep(0.3)
        self.home()

    def shake(self):
        print("[ROARM-HW] Shaking Head (No)")
        for _ in range(2):
            self.send_cmd({"T": 101, "joint": 1, "rad": 0.4, "spd": 250})
            time.sleep(0.3)
            self.send_cmd({"T": 101, "joint": 1, "rad": -0.4, "spd": 250})
            time.sleep(0.3)
        self.home()

    def pick(self):
        print("[ROARM-HW] Pick Object")
        self.set_gripper(1.0)
        time.sleep(0.5)
        self.move_xyz(200, 0, 50)
        time.sleep(1.0)
        self.set_gripper(0.0)
        time.sleep(0.5)
        self.move_xyz(200, 0, 150)
        time.sleep(0.8)

    def place(self):
        print("[ROARM-HW] Place Object")
        self.move_xyz(150, 150, 120)
        time.sleep(1.0)
        self.move_xyz(150, 150, 50)
        time.sleep(0.8)
        self.set_gripper(1.0)
        time.sleep(0.5)
        self.home()

    def execute_action(self, action_name):
        action_map = {
            "wave": self.wave,
            "nod_head": self.nod,
            "shake_head": self.shake,
            "pick_object": self.pick,
            "place_object": self.place,
            "reset_to_home": self.home
        }
        fn = action_map.get(action_name)
        if fn:
            fn()
        else:
            print(f"[ROARM-HW] Unknown action: {action_name}")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


if __name__ == "__main__":
    print("\nScanning available serial ports on this Mac:")
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("  No serial devices detected. (Plug in RoArm-M2 USB to test).")
    else:
        for p in ports:
            print(f"  • {p.device}: {p.description}")

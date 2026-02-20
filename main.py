import argparse

from config import Config
from driver_monitor import DriverMonitor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="Video file path or camera index (default: 0)")
    args = parser.parse_args()

    cfg = Config()
    source = args.source if args.source is not None else cfg.CAMERA_INDEX
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    monitor = DriverMonitor(cfg)
    try:
        monitor.start(source)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")


if __name__ == "__main__":
    main()

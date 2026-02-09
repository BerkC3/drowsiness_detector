from config import Config
from driver_monitor import DriverMonitor


def main():
    monitor = DriverMonitor(Config())
    try:
        monitor.start()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")


if __name__ == "__main__":
    main()

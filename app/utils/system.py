import psutil
import platform


def get_system_metrics():
    return {
        "CPU": platform.processor(),
        "RAM_Total_GB": round(psutil.virtual_memory().total / (1024**3), 2),
        "OS": platform.system(),
    }

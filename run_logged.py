"""
run_logged.py

هر اسکریپتی رو به‌جای اجرای مستقیم، از طریق این فایل اجرا کن.
خروجی همون‌طور زنده توی ترمینال چاپ میشه (چیزی از دست نمیره)
و هم‌زمان با نام فایل، تاریخ/ساعت شروع و پایان، مدت اجرا و exit code
داخل یک فایل لاگ ثابت append میشه.

Usage:
    python run_logged.py <script.py> [args...]

Example:
    python run_logged.py step32_graphsage_15seed_results.py
    python run_logged.py experiment_learnable_adjacency_supervised_test.py --seed 42
"""

import subprocess
import sys
import datetime
from pathlib import Path

LOG_FILE = Path(__file__).parent / "run_log.txt"
SEPARATOR = "=" * 80


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_logged.py <script.py> [args...]")
        sys.exit(1)

    script_path = sys.argv[1]
    script_args = sys.argv[2:]

    if not Path(script_path).exists():
        print(f"خطا: فایل '{script_path}' پیدا نشد.")
        sys.exit(1)

    start_time = datetime.datetime.now()
    header = (
        f"\n{SEPARATOR}\n"
        f"SCRIPT : {script_path}\n"
        f"ARGS   : {' '.join(script_args) if script_args else '(none)'}\n"
        f"START  : {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{SEPARATOR}\n"
    )

    print(header, end="")

    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(header)
        log.flush()

        cmd = [sys.executable, "-u", script_path] + script_args
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # خروجی زنده رو هم‌زمان توی ترمینال چاپ کن و توی فایل بنویس
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()

        process.wait()
        end_time = datetime.datetime.now()
        duration = end_time - start_time

        footer = (
            f"\n{'-' * 80}\n"
            f"END       : {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"DURATION  : {duration}\n"
            f"EXIT CODE : {process.returncode}\n"
            f"{SEPARATOR}\n\n"
        )
        print(footer, end="")
        log.write(footer)

    sys.exit(process.returncode)


if __name__ == "__main__":
    main()

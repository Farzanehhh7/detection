"""
run_pipeline.py

اجرای مرتب یک لیست از اسکریپت‌ها به ترتیب، با append کردن خروجی همه‌شون
به یک فایل لاگ مشترک (run_log.txt) — همون فرمت run_logged.py، به‌علاوه‌ی
این‌که اگه یکی از مراحل fail بشه، پایپ‌لاین متوقف میشه (چون مرحله‌ی بعدی
معمولاً به فایل خروجی مرحله‌ی قبلی وابسته‌ست).

یک فایل متنی به اسم pipeline_order.txt کنارش بساز، هر خط یک مسیر اسکریپت
به ترتیب اجرا. خط‌هایی که با # شروع می‌شن یا خالی‌ان نادیده گرفته می‌شن:

    # preprocessing
    src/01_preprocessing/clean_elliptic.py
    src/01_preprocessing/clean_samld.py
    # graph construction
    src/02_graph_construction/build_graph.py
    # model + training
    src/04_training/train_final_model.py
    # evaluation
    src/05_evaluation/multi_seed_eval.py

Usage:
    python run_pipeline.py pipeline_order.txt
    python run_pipeline.py pipeline_order.txt --continue-on-error
"""

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

LOG_FILE = Path(__file__).parent / "run_log.txt"
SEPARATOR = "=" * 80


def read_pipeline(order_file: Path) -> list[str]:
    scripts = []
    for line in order_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        scripts.append(line)
    return scripts


def run_one(script_path: str, log) -> int:
    start_time = datetime.datetime.now()
    header = (
        f"\n{SEPARATOR}\n"
        f"SCRIPT : {script_path}\n"
        f"START  : {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{SEPARATOR}\n"
    )
    print(header, end="")
    log.write(header)
    log.flush()

    process = subprocess.Popen(
        [sys.executable, "-u", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        log.write(line)
        log.flush()
    process.wait()

    end_time = datetime.datetime.now()
    footer = (
        f"\n{'-' * 80}\n"
        f"END       : {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"DURATION  : {end_time - start_time}\n"
        f"EXIT CODE : {process.returncode}\n"
        f"{SEPARATOR}\n\n"
    )
    print(footer, end="")
    log.write(footer)
    return process.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("order_file", help="فایل متنی شامل ترتیب اجرای اسکریپت‌ها")
    parser.add_argument("--continue-on-error", action="store_true",
                         help="با خطا هم بقیه‌ی مراحل رو ادامه بده (پیش‌فرض: متوقف میشه)")
    args = parser.parse_args()

    scripts = read_pipeline(Path(args.order_file))
    if not scripts:
        print("هیچ اسکریپتی توی فایل ترتیب پیدا نشد.")
        sys.exit(1)

    print(f"{len(scripts)} مرحله برای اجرا پیدا شد.\n")

    with open(LOG_FILE, "a", encoding="utf-8") as log:
        run_header = (
            f"\n{'#' * 80}\n"
            f"PIPELINE RUN START: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'#' * 80}\n"
        )
        print(run_header, end="")
        log.write(run_header)

        for i, script in enumerate(scripts, 1):
            print(f"[{i}/{len(scripts)}] در حال اجرا: {script}")
            exit_code = run_one(script, log)
            if exit_code != 0 and not args.continue_on_error:
                msg = f"\n❌ مرحله‌ی '{script}' با exit code {exit_code} متوقف شد. پایپ‌لاین قطع شد.\n"
                print(msg)
                log.write(msg)
                sys.exit(exit_code)

        run_footer = (
            f"\n{'#' * 80}\n"
            f"PIPELINE RUN COMPLETE: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'#' * 80}\n"
        )
        print(run_footer, end="")
        log.write(run_footer)


if __name__ == "__main__":
    main()

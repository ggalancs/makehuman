#!/usr/bin/env python3
"""
Run MakeHuman end to end in an isolated home with tests/e2e/smoke_plugin.py.

    .venv/bin/python tests/e2e/run_smoke.py [--timeout 600] [--keep] [--only check1,check2]

The app is started from a different working directory than its own (that is
part of what is tested), with MH_HOME_LOCATION pointing at a fresh temporary
folder, so the user's real MakeHuman settings and files are never touched.

The wall-clock limit is enforced here, from outside the app: the watchdog only
knows a PID and a clock, so a hung GUI cannot stop it from firing.

Exit status is 0 only when every check passed, the run completed and the app
exited by itself with status 0.
"""

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MH_DIR = os.path.join(REPO, "makehuman")
PLUGIN_NAME = "9_zz_smoketest"


def read_version_info():
    with open(os.path.join(MH_DIR, "makehuman.py"), encoding="utf-8") as f:
        version = re.search(r'^__version__ = "([^"]+)"', f.read(), re.M).group(1)
    with open(os.path.join(MH_DIR, "core", "mhmain.py"), encoding="utf-8") as f:
        sentinel = re.search(r"'_versionSentinel': '([0-9A-F]+)'", f.read()).group(1)
    return version, sentinel


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def prepare_home(home, port):
    """Create an isolated MakeHuman user folder with the smoke plugin enabled."""
    user_dir = os.path.join(home, "Documents", "MakeHuman", "v1py3")
    plugins = os.path.join(user_dir, "plugins")
    os.makedirs(plugins)
    shutil.copy(os.path.join(HERE, "smoke_plugin.py"), os.path.join(plugins, PLUGIN_NAME + ".py"))
    version, sentinel = read_version_info()
    settings = {"version": version, "_versionSentinel": sentinel, "activeUserPlugins": [PLUGIN_NAME]}
    with open(os.path.join(user_dir, "settings.ini"), "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4)
    with open(os.path.join(user_dir, "socket.cfg"), "w", encoding="utf-8") as f:
        json.dump({"acceptConnections": True, "advanced": False, "host": "127.0.0.1", "port": port}, f)
    return user_dir


def run(python, timeout, only=None, keep=False, out=sys.stdout, coverage=False):
    home = tempfile.mkdtemp(prefix="mh-e2e-")
    port = free_port()
    user_dir = prepare_home(home, port)
    results_path = os.path.join(home, "results.json")
    log_path = os.path.join(home, "app.log")
    env = dict(os.environ, MH_HOME_LOCATION=home, MH_SMOKE_RESULTS=results_path,
               MH_SMOKE_OUTDIR=os.path.join(home, "out"), MH_SMOKE_SOCKET_PORT=str(port),
               PYTHONUNBUFFERED="1")
    # The app needs the real display (tests/conftest.py sets offscreen for unit tests)
    env.pop("QT_QPA_PLATFORM", None)
    if only:
        env["MH_SMOKE_ONLY"] = only

    start = time.time()
    with open(log_path, "wb") as log:
        cmd = [python, "-u", os.path.join(MH_DIR, "makehuman.py")]
        if coverage:
            cmd[1:2] = ["-m", "coverage", "run", "--data-file", os.path.join(home, ".coverage"),
                        "--source", MH_DIR, "--omit", "*/data/*,*/testsuite/*," + PLUGIN_NAME + ".py"]
        proc = subprocess.Popen(cmd, cwd=home, env=env, stdout=log, stderr=subprocess.STDOUT)
        timed_out = False
        while proc.poll() is None:
            if time.time() - start > timeout:
                timed_out = True
                proc.terminate()
                try:
                    proc.wait(10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                break
            time.sleep(1)
    elapsed = time.time() - start

    data = {"complete": False, "results": [], "modals": []}
    if os.path.isfile(results_path):
        with open(results_path, encoding="utf-8") as f:
            data = json.load(f)

    failed = [r for r in data["results"] if not r.get("ok")]
    for r in data["results"]:
        status = "RUNNING (hung here)" if r.get("running") else ("ok  " if r.get("ok") else "FAIL")
        print("%-5s %-26s %6.1fs  %s" % (status, r["name"], r.get("seconds", 0),
                                          json.dumps(r.get("detail"))[:110] if r.get("ok") else ""), file=out)
        if not r.get("ok"):
            for key in ("error", "logged_errors"):
                if r.get(key):
                    text = r[key] if isinstance(r[key], str) else "\n".join(r[key])
                    print("      " + text.strip().replace("\n", "\n      ")[:4000], file=out)
    if data.get("modals"):
        print("modal dialogs dismissed: %s" % data["modals"], file=out)

    problems = []
    if not data["results"]:
        problems.append("no results written (the plugin never ran)")
    if timed_out:
        problems.append("timed out after %ds" % timeout)
    elif proc.returncode != 0:
        problems.append("app exit status %s" % proc.returncode)
    if not data["complete"]:
        problems.append("run incomplete")
    if failed:
        problems.append("%d/%d checks failed" % (len(failed), len(data["results"])))

    if coverage and os.path.isfile(os.path.join(home, ".coverage")):
        report = subprocess.run([python, "-m", "coverage", "report", "--data-file", os.path.join(home, ".coverage"),
                                 "--sort=cover"], capture_output=True, text=True)
        print(report.stdout[-3000:], file=out)
    print("%s in %.0fs. home: %s" % ("PASSED" if not problems else "FAILED: " + "; ".join(problems),
                                     elapsed, home), file=out)
    if not problems and not keep:
        shutil.rmtree(home)  # created above by mkdtemp
    return 0 if not problems else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--timeout", type=int, default=600, help="hard wall-clock limit in seconds")
    parser.add_argument("--keep", action="store_true", help="keep the temporary home even when passing")
    parser.add_argument("--only", help="comma separated check names to run")
    parser.add_argument("--python", default=sys.executable, help="interpreter that runs MakeHuman")
    parser.add_argument("--coverage", action="store_true", help="measure which application lines the run executes")
    args = parser.parse_args(argv)
    return run(args.python, args.timeout, args.only, args.keep, coverage=args.coverage)


if __name__ == "__main__":
    sys.exit(main())

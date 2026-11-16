"""Start the supplied Spring backend against the existing Python policy server."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--java", default="java")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    from policy_r1.runtime_env import load_runtime_env
    load_runtime_env(ROOT/".env")
    if not os.environ.get("AI_INTERNAL_KEY", "").strip():
        parser.error("Set AI_INTERNAL_KEY in the workspace .env")
    source = ROOT/"integrations/am-erica-backend"
    jars = [p for p in (source/"build/libs").glob("*.jar") if not p.name.endswith("-plain.jar")]
    if len(jars) != 1:
        parser.error("Build the backend bootJar first")
    environment = dict(os.environ)
    environment.pop("SERPAPI_API_KEY", None)  # Only Python needs the provider credential.
    environment.update(SPRING_PROFILES_ACTIVE="local,policy-mvp", POLICY_REMINDER_ENABLED="false")
    command = [args.java, "-jar", str(jars[0]), "--server.address=127.0.0.1", f"--server.port={args.port}",
               "--ai.mock.seed-demo-data=false"]
    os.chdir(source)
    os.execvpe(args.java, command, environment)


if __name__ == "__main__":
    main()

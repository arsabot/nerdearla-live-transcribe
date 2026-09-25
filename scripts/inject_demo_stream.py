#!/usr/bin/env python3
"""
scripts/inject_demo_stream.py
Interactive CLI Data Injector for Nerdearla Live Demo.
Streams realistic speech transcripts and interim tokens to conference stages in real time.
"""

import argparse
import asyncio
import json
import sys
import time
from typing import Dict, List, Optional
import urllib.request
import urllib.error

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

STAGE_SCRIPTS: Dict[str, List[Dict[str, str]]] = {
    "stage-a": [
        {
            "interim": "Welcome everyone to Nerdearla",
            "final": "Welcome everyone to Nerdearla 2026 AI and Open Source stage!",
        },
        {
            "interim": "Today we are discussing running local neural networks",
            "final": "Today we are discussing running local neural networks directly in the browser using WebGPU and ONNX Runtime.",
        },
        {
            "interim": "Our open source real-time STT engine",
            "final": "Our open source real-time STT engine eliminates latency without compromising attendee audio privacy.",
        },
        {
            "interim": "We integrated Google Gemini Live",
            "final": "We integrated Google Gemini Live for sub-second simultaneous translation with automated Google Translate fallback.",
        },
        {
            "interim": "Please submit your pull request",
            "final": "Please submit your pull request to the GitHub repository after this session!",
        },
    ],
    "stage-b": [
        {
            "interim": "Good morning and welcome to the Cloud Infrastructure stage",
            "final": "Good morning and welcome to the Cloud Infrastructure & DevOps stage at Nerdearla!",
        },
        {
            "interim": "Managing multi-region Kubernetes clusters",
            "final": "Managing multi-region Kubernetes clusters requires robust observability and automated circuit breakers.",
        },
        {
            "interim": "Each container in our deployment pipeline",
            "final": "Each container in our deployment pipeline is automatically scanned for security vulnerabilities.",
        },
        {
            "interim": "The load balancer distributes traffic",
            "final": "The load balancer distributes traffic evenly across pods to maintain ultra-low latency.",
        },
        {
            "interim": "Thank you for joining our DevOps workshop",
            "final": "Thank you for joining our DevOps workshop today at Nerdearla!",
        },
    ],
    "stage-c": [
        {
            "interim": "Welcome to the Web Architecture track",
            "final": "Welcome to the Web Architecture track! Today we explore WebAssembly and real-time WebSockets.",
        },
        {
            "interim": "Browser audio worklets process 16 kHz PCM",
            "final": "Browser audio worklets process 16 kHz PCM audio chunks without blocking the UI rendering thread.",
        },
    ],
    "stage-d": [
        {
            "interim": "Security is fundamental in zero trust environments",
            "final": "Security is fundamental in zero trust environments: cryptographic HMAC cookie signatures prevent tampering.",
        },
    ],
}


def send_inject_request(host: str, stage: str, msg_type: str, text: str, token: Optional[str] = None) -> Optional[dict]:
    url = f"{host.rstrip('/')}/api/session/{stage}/inject"
    payload = json.dumps({
        "type": msg_type,
        "text": text,
        "source_language": "en",
        "audio_timestamp": time.time() * 1000,
        "client_sent_ms": time.time() * 1000,
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    try:
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            elapsed_ms = (time.perf_counter() - t0) * 1000
            data["_client_latency_ms"] = round(elapsed_ms, 1)
            return data
    except urllib.error.HTTPError as e:
        print(f"  [ERROR] HTTP {e.code} on {stage}: {e.read().decode('utf-8')[:120]}")
        return None
    except Exception as e:
        print(f"  [ERROR] Connection failed on {stage}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Inject live speech transcripts into Nerdearla Live sessions.")
    parser.add_argument("--host", default="http://localhost:3000", help="Base URL of Nerdearla Live (default: http://localhost:3000)")
    parser.add_argument("--stage", default="both", choices=["both", "all", "stage-a", "stage-b", "stage-c", "stage-d"], help="Target stage (default: both)")
    parser.add_argument("--interval", type=float, default=4.0, help="Seconds between statements (default: 4.0)")
    parser.add_argument("--interim", action="store_true", default=True, help="Send interim preview before finalized phrase (default: True)")
    parser.add_argument("--loops", type=int, default=0, help="Number of loops to run (0 = continuous infinite stream)")
    parser.add_argument("--token", default="", help="Optional Bearer auth token if AUTH_ENABLED=true")
    parser.add_argument("--text", default="", help="Custom text to inject immediately (one-shot mode)")
    parser.add_argument("--lang", default="en", help="Source language for custom text (default: en)")
    args = parser.parse_args()

    stages = ["stage-a", "stage-b"] if args.stage == "both" else (list(STAGE_SCRIPTS.keys()) if args.stage == "all" else [args.stage])

    print("=" * 65)
    print("  🎙️ Nerdearla Live — Multi-Stage Real-Time Data Injector")
    print(f"  Target Host: {args.host}")
    print(f"  Stages:      {', '.join(stages)}")
    if args.text:
        print(f"  Mode:        Custom One-Shot Injection")
        print(f"  Text:        \"{args.text}\" (lang: {args.lang})")
    else:
        print(f"  Mode:        Automated Multi-Stage Stream ({'Infinite' if args.loops == 0 else f'{args.loops} loops'})")
        print(f"  Interval:    {args.interval}s")
    print("=" * 65)

    if args.text:
        for stage in stages:
            print(f"\n📡 [{stage.upper()}] Injecting Custom Statement:")
            if args.interim:
                words = args.text.split()
                if len(words) > 3:
                    interim_sample = " ".join(words[: len(words) // 2])
                    print(f"   ⏳ Interim: \"{interim_sample}\"")
                    send_inject_request(args.host, stage, "interim", interim_sample, args.token)
                    time.sleep(0.8)
            print(f"   🎙️ Final:   \"{args.text}\"")
            res = send_inject_request(args.host, stage, "final", args.text, args.token)
            if res and "event" in res:
                ev = res["event"]
                es = ev.get("translations", {}).get("es", "")
                pt = ev.get("translations", {}).get("pt", "")
                metrics = ev.get("metrics", {})
                tot_ms = metrics.get("total_ms", res.get("_client_latency_ms", 0))
                print(f"   🇪🇸 ES:     \"{es}\"")
                if pt:
                    print(f"   🇧🇷 PT:     \"{pt}\"")
                print(f"   ⚡ Latency: {tot_ms} ms (Delivery to {res.get('delivered_viewers', 0)} viewers)")
        print("\n✅ Custom statement successfully injected into all target stages.")
        return

    step = 0
    loop_count = 0
    try:
        while True:
            loop_count += 1
            print(f"\n--- [Cycle #{loop_count}] Streaming phrases to stages ---")
            for stage in stages:
                script = STAGE_SCRIPTS.get(stage, STAGE_SCRIPTS["stage-a"])
                item = script[step % len(script)]

                print(f"\n📡 [{stage.upper()}] Statement #{step + 1}:")
                if args.interim and item.get("interim"):
                    print(f"   ⏳ Interim: \"{item['interim']}\"")
                    send_inject_request(args.host, stage, "interim", item["interim"], args.token)
                    time.sleep(1.2)

                print(f"   🎙️ Final:   \"{item['final']}\"")
                res = send_inject_request(args.host, stage, "final", item["final"], args.token)
                if res and "event" in res:
                    ev = res["event"]
                    es = ev.get("translations", {}).get("es", "")
                    pt = ev.get("translations", {}).get("pt", "")
                    metrics = ev.get("metrics", {})
                    tot_ms = metrics.get("total_ms", res.get("_client_latency_ms", 0))
                    print(f"   🇪🇸 ES:     \"{es}\"")
                    if pt:
                        print(f"   🇧🇷 PT:     \"{pt}\"")
                    print(f"   ⚡ Latency: {tot_ms} ms (Delivery to {res.get('delivered_viewers', 0)} viewers)")

            step += 1
            if args.loops > 0 and loop_count >= args.loops:
                print("\n✅ Specified loop count completed.")
                break

            print(f"\n⏳ Waiting {args.interval}s before next speech statement... (Ctrl+C to stop)")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n🛑 Stream injection stopped by user.")


if __name__ == "__main__":
    main()

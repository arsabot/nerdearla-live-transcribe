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
    "demo-stage-a": [
        {
            "interim": "Welcome everyone to Nerdearla",
            "final": "Welcome everyone to Nerdearla 2026 AI and Open Source stage!",
        },
        {
            "interim": "Today we are discussing running local neural networks",
            "final": "Today we are discussing running local neural networks directly in the browser using WebGPU and ONNX Runtime.",
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
    "demo-stage-b": [
        {
            "interim": "Olá a todos e bem-vindos ao Nerdearla",
            "final": "Olá a todos e bem-vindos ao Nerdearla 2026 na trilha de Cloud e DevOps!",
        },
        {
            "interim": "Gerenciar clusters de Kubernetes em produção",
            "final": "Gerenciar clusters de Kubernetes em produção exige alta observabilidade e resiliência automatizada.",
        },
        {
            "interim": "Nossa arquitetura em nuvem distribui o tráfego",
            "final": "Nossa arquitetura em nuvem distribui o tráfego de maneira eficiente mantendo ultra-baixa latência.",
        },
        {
            "interim": "Muito obrigado a todos os desenvolvedores",
            "final": "Muito obrigado a todos os desenvolvedores por participarem deste workshop no Nerdearla!",
        },
    ],
    "demo-stage-c": [
        {
            "interim": "Bienvenidos a la charla de arquitectura en Nerdearla",
            "final": "¡Bienvenidos a la charla de arquitectura y código abierto en Nerdearla 2026!",
        },
        {
            "interim": "Esta plataforma procesa transcripción y traducción",
            "final": "Esta plataforma procesa transcripción y traducción simultánea en tiempo real con latencia sub-segundo.",
        },
        {
            "interim": "Utilizamos WebSockets y WebGPU en el navegador",
            "final": "Utilizamos WebSockets y WebGPU en el navegador para eliminar intermediarios y garantizar privacidad.",
        },
        {
            "interim": "Muchas gracias a toda la comunidad",
            "final": "¡Muchas gracias a toda la comunidad de Nerdearla por acompañarnos hoy en esta demo!",
        },
    ],
    "stage-a": [
        {
            "interim": "Welcome everyone to Nerdearla",
            "final": "Welcome everyone to Nerdearla 2026 AI and Open Source stage!",
        },
    ],
    "stage-b": [
        {
            "interim": "Good morning and welcome to the Cloud Infrastructure stage",
            "final": "Good morning and welcome to the Cloud Infrastructure & DevOps stage at Nerdearla!",
        },
    ],
    "stage-c": [
        {
            "interim": "Welcome to the Web Architecture track",
            "final": "Welcome to the Web Architecture track! Today we explore WebAssembly and real-time WebSockets.",
        },
    ],
    "stage-d": [
        {
            "interim": "Security is fundamental in zero trust environments",
            "final": "Security is fundamental in zero trust environments: cryptographic HMAC cookie signatures prevent tampering.",
        },
    ],
}

STAGE_SOURCE_LANGS: Dict[str, str] = {
    "demo-stage-a": "en",
    "demo-stage-b": "pt",
    "demo-stage-c": "es",
    "stage-a": "en",
    "stage-b": "en",
    "stage-c": "en",
    "stage-d": "en",
}


def send_inject_request(host: str, stage: str, msg_type: str, text: str, token: Optional[str] = None, source_lang: Optional[str] = None) -> Optional[dict]:
    url = f"{host.rstrip('/')}/api/session/{stage}/inject"
    lang = source_lang or STAGE_SOURCE_LANGS.get(stage, "en")
    payload = json.dumps({
        "type": msg_type,
        "text": text,
        "source_language": lang,
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
    parser.add_argument(
        "--stage",
        default="demo-all",
        choices=["demo-all", "demo-stage-a", "demo-stage-b", "demo-stage-c", "stage-a", "stage-b", "stage-c", "stage-d", "both", "all"],
        help="Target stage (default: demo-all, strictly isolated 3 demo rooms)",
    )
    parser.add_argument("--interval", type=float, default=4.0, help="Seconds between statements (default: 4.0)")
    parser.add_argument("--interim", action="store_true", default=True, help="Send interim preview before finalized phrase (default: True)")
    parser.add_argument("--loops", type=int, default=0, help="Number of loops to run (0 = continuous infinite stream)")
    parser.add_argument("--token", default="", help="Optional Bearer auth token if AUTH_ENABLED=true")
    parser.add_argument("--text", default="", help="Custom text to inject immediately (one-shot mode)")
    parser.add_argument("--lang", default="", help="Optional source language override for custom text")
    args = parser.parse_args()

    if args.stage == "demo-all":
        stages = ["demo-stage-a", "demo-stage-b", "demo-stage-c"]
        is_isolated_sandbox = True
    elif args.stage in ("demo-stage-a", "demo-stage-b", "demo-stage-c"):
        stages = [args.stage]
        is_isolated_sandbox = True
    elif args.stage == "both":
        stages = ["stage-a", "stage-b"]
        is_isolated_sandbox = False
    elif args.stage == "all":
        stages = ["stage-a", "stage-b", "stage-c", "stage-d"]
        is_isolated_sandbox = False
    else:
        stages = [args.stage]
        is_isolated_sandbox = args.stage.startswith("demo-")

    print("=" * 68)
    print("  🎙️ Nerdearla Live — Multi-Stage Real-Time Data Injector")
    print(f"  Target Host: {args.host}")
    print(f"  Stages:      {', '.join(stages)}")
    if is_isolated_sandbox:
        print("  🛡️  ISOLATION: Streaming to 3 Isolated Demo Sandboxes.")
        print("     - demo-stage-a: EN -> ES, PT")
        print("     - demo-stage-b: PT -> EN, ES")
        print("     - demo-stage-c: ES -> EN, PT")
        print("     Real conference stages (stage-a, stage-b, etc.) are PROTECTED.")
    else:
        print("  ⚠️  ATTENTION: Streaming to PRODUCTION STAGE(S)!")
        print("     Real attendees will see these subtitles.")
    if args.text:
        print(f"  Mode:        Custom One-Shot Injection")
        print(f"  Text:        \"{args.text}\" (lang: {args.lang})")
    else:
        print(f"  Mode:        Automated Multi-Stage Stream ({'Infinite' if args.loops == 0 else f'{args.loops} loops'})")
        print(f"  Interval:    {args.interval}s")
    print("=" * 68)

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
                src = ev.get("source_language", "").upper()
                trans = ev.get("translations", {})
                metrics = ev.get("metrics", {})
                tot_ms = metrics.get("total_ms", res.get("_client_latency_ms", 0))
                for lang, trans_text in trans.items():
                    if lang.lower() != src.lower() and trans_text:
                        flag = "🇬🇧 EN" if lang.lower() == "en" else ("🇪🇸 ES" if lang.lower() == "es" else ("🇧🇷 PT" if lang.lower() == "pt" else lang.upper()))
                        print(f"   {flag}:  \"{trans_text}\"")
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
                script = STAGE_SCRIPTS.get(stage, STAGE_SCRIPTS["demo-stage-a"])
                item = script[step % len(script)]
                src_lang = STAGE_SOURCE_LANGS.get(stage, "en").upper()

                print(f"\n📡 [{stage.upper()}] (Origin: {src_lang}) Statement #{step + 1}:")
                if args.interim and item.get("interim"):
                    print(f"   ⏳ Interim: \"{item['interim']}\"")
                    send_inject_request(args.host, stage, "interim", item["interim"], args.token)
                    time.sleep(1.2)

                print(f"   🎙️ Final:   \"{item['final']}\"")
                res = send_inject_request(args.host, stage, "final", item["final"], args.token)
                if res and "event" in res:
                    ev = res["event"]
                    src = ev.get("source_language", "").upper()
                    trans = ev.get("translations", {})
                    metrics = ev.get("metrics", {})
                    tot_ms = metrics.get("total_ms", res.get("_client_latency_ms", 0))
                    for lang, trans_text in trans.items():
                        if lang.lower() != src.lower() and trans_text:
                            flag = "🇬🇧 EN" if lang.lower() == "en" else ("🇪🇸 ES" if lang.lower() == "es" else ("🇧🇷 PT" if lang.lower() == "pt" else lang.upper()))
                            print(f"   {flag}:  \"{trans_text}\"")
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

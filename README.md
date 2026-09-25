# Nerdearla Live 🎙️🌐

> **Open Source Real-Time Accessibility & Simultaneous Translation for Conferences**  
> *Plataforma de Accesibilidad y Traducción Simultánea en Tiempo Real para Conferencias*  
> **Built for the Nerdearla Vibeathon 2026 Challenge**

[![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/arsabot/nerdearla-live-transcribe)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12+-green.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](docker-compose.yml)
[![Tests](https://img.shields.io/badge/Tests-66%20Passing-brightgreen.svg)](tests/)
[![WebGPU](https://img.shields.io/badge/WebGPU-Whisper%20%26%20Nemotron-purple.svg)](app/static/)
[![Gemini Live](https://img.shields.io/badge/Gemini%20Live-WebSocket%20Bidi-orange.svg)](app/gemini_live.py)

---

### 🌐 Idioma / Language
- 🇪🇸 **[Leer en Español](#-versión-en-español)**
- 🇬🇧 **[Read in English](#-english-version)**

---

# 🇪🇸 Versión en Español

## 📑 Tabla de Contenidos
- [🎯 ¿Qué problema soluciona?](#-qué-problema-soluciona)
- [📖 ¿Qué es Nerdearla Live?](#-qué-es-nerdearla-live)
- [🏗️ Arquitectura del Sistema](#-arquitectura-del-sistema)
- [✨ Características Principales](#-características-principales)
- [🗺️ Mapa de URLs y Roles](#-mapa-de-urls-del-sistema)
- [🚀 Guía de Inicio Rápido](#-guía-de-inicio-rápido)
  - [Opción 1: Docker Compose (Recomendado)](#opción-1-ejecutar-con-docker-compose-recomendado)
  - [Opción 2: Entorno Local Python](#opción-2-ejecutar-localmente-con-python)
- [⚙️ Configuración de Variables de Entorno (.env)](#-configuración-de-variables-de-entorno-env)
- [🧪 Pruebas Automatizadas y Calidad](#-ejecución-de-pruebas-automatizadas)

---

## 🎯 ¿Qué problema soluciona?

En conferencias masivas de tecnología como **Nerdearla** (con más de 10.000 asistentes y decenas de charlas simultáneas), la comunicación enfrenta barreras críticas:

1. **Barrera Idiomática y Exclusión**:
   - Oradores internacionales dan sus charlas en **inglés o portugués**, mientras gran parte de la audiencia local prefiere o necesita seguir la presentación en **español** (o viceversa).
   - Contratar cabinas y receptores de interpretación humana para 5, 10 o 30 escenarios en simultáneo es **prohibitivamente costoso** e inmanejable a gran escala.
2. **Falta de Accesibilidad Real (Personas con Hipoacusia)**:
   - Asistentes sordos o con dificultades auditivas quedan completamente excluidos si no disponen de transcripción textual inmediata.
   - En auditorios ruidosos o en streaming, seguir tecnicismos complejos de viva voz resulta fatigante y propenso a pérdidas de contexto.
3. **Fallas Críticas en Nombres Propios y Términos Técnicos**:
   - Los motores de transcripción genéricos confunden sistemáticamente *"Nerdearla"* con frases erróneas (*"nerd de habla"*, *"nerd arla"*, *"merdearla"*), y destruyen la jerga técnica (*Kubernetes, Pull Requests, DevOps, Open Source*).
4. **Costos Astronómicos de Servidor (GPU) y Privacidad del Audio**:
   - Procesar decenas de canales de audio continuo en servidores cloud acumula facturas enormes de GPU o expone el audio confidencial de los oradores a terceros.

**Nerdearla Live soluciona todo esto de raíz**:
- **Cero costo de GPU en el servidor**: Gracias a la inferencia local en el navegador del orador vía **WebGPU** (Whisper y Nemotron) o la Web Speech API, el audio nunca sale del dispositivo para transcribirse.
- **Traducción simultánea en < 700 ms**: Conexión bidireccional por WebSocket a **Gemini Live API** con conmutación por error automática e instantánea a Google Translate.
- **Speech-Biasing & Normalización para "Nerdearla"**: Glosario acústico y fonético que corrige distorsiones en tiempo real.
- **Orquestación Multi-Sala Aislada**: Reparto instantáneo de subtítulos a teléfonos de la audiencia, pantallas gigantes y sistemas de streaming con salas independientes.

---

## 📖 ¿Qué es Nerdearla Live?

**Nerdearla Live** es una plataforma integral, autosoportada (*self-hosted*) y de código abierto para la accesibilidad y traducción simultánea de conferencias.

Permite que cualquier orador abra un enlace en su laptop o celular (`/speaker/stage-a`), active su micrófono, y transmita subtítulos multilingües sincronizados a la audiencia (`/session/stage-a`) y a la pantalla del escenario (`/stage/stage-a`) con latencia imperceptible.

---

## 🏗️ Arquitectura del Sistema

```text
                                  NERDEARLA 2026
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
          STAGE A                    STAGE B                    STAGE N
      AI & Open Source           Cloud & DevOps               Security
             │                          │                          │
             ▼                          ▼                          ▼
    MICROPHONE + VISUALIZER    MICROPHONE + VISUALIZER    MANUAL INJECTION
   (/speaker/stage-a WebGPU)  (/speaker/stage-b Cloud)    (Staff Announcements)
             │                          │                          │
             ▼                          ▼                          ▼
      Speech-to-Text             Speech-to-Text             Manual Text
    • Whisper Local (WebGPU)   • Deepgram Nova-3          • Q&A Prompts
    • Nemotron-3.5 (WebGPU)    • ElevenLabs Scribe        • Sponsor Mentions
    • Web Speech API           • Web Speech API                    │
             │                          │                          │
             └──────────────────────────┼──────────────────────────┘
                                        ▼
                   PIPELINE DE TRADUCCIÓN REALTIME + GLOSARIO
           ┌────────────────────────────────────────────────────────┐
           │ • Conexión WebSocket Bidireccional Gemini Live API     │
           │ • Fallback Instantáneo y Automático a Google Translate │
           │ • Normalización Fonética ("Nerdearla" + Tech Glossary) │
           └────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                         SESSION MANAGER MULTI-SALA
                        (Aislamiento Estricto por Sala)
                                        │
     ┌──────────────────────────────────┼──────────────────────────────────┐
     ▼                                  ▼                                  ▼
VISTA AUDIENCIA                  PANTALLA ESCENARIO                SALA DE CONTROL
(/session/{id})                  (/stage/{id})                     (/producer)
• Subtítulos grandes             • Modo proyector alto contraste   • Métricas de latencia en vivo
• Selector de idioma             • Compatible con OBS / vMix       • Monitoreo de espectadores
• Modo Claro / Oscuro            • Sin controles molestos          • Borrado de historial en vivo
• Exportación .SRT/.VTT                                            • Inyección de subtítulos
```

---

## ✨ Características Principales

1. **Orquestación Multi-Sala (Multi-Stage)**:
   - Gestión simultánea de **2, 5, 10 o 30+ escenarios** (`Stage A`, `Stage B`, etc.) con aislamiento estricto de eventos por WebSocket.
   - Creación, edición, pausa y eliminación de salas en tiempo real desde el panel de control.
2. **5 Motores de Reconocimiento de Voz (STT)**:
   - **Whisper Local (WebGPU / WASM)**: Corre en el navegador del orador vía Transformers.js. Audio 100% privado y costo cero en GPU de servidor.
   - **Nemotron Local (ONNX Runtime Web)**: Modelo de streaming NVIDIA Nemotron-3.5-ASR acelerado por WebGPU en el cliente.
   - **Web Speech API**: STT nativo del navegador sin sobrecarga de backend.
   - **Deepgram Nova-3**: STT en streaming en la nube con puntuación y formato de alta precisión.
   - **ElevenLabs Scribe v2**: Reconocimiento en tiempo real vía WebSocket.
3. **Traducción Realtime con Gemini Live & Fallback Automático**:
   - Conexión directa bidireccional por WebSocket a **Gemini Live API** (`gemini-3.5-live-translate-preview` / `gemini-2.0-flash-exp`).
   - Circuit breaker con failover transparente a **Google Translate** ante cortes de red o límites de cuota, y recuperación automática.
4. **Detección y Speech-Biasing para "Nerdearla"**:
   - Normalización automática de errores fonéticos habituales (*"nerd de habla"*, *"nerd arla"*, *"nerdear la"*, *"merdearla"*, *"nerderla"* -> `Nerdearla`).
   - Priorización acústica de términos y glosario técnico (*Kubernetes, Pull Request, DevOps, Open Source*).
5. **Monitor de Micrófono & Visualizador de Audio en Vivo**:
   - Indicador visual dinámico de decibelios y frecuencia mediante `AudioContext` en el micrófono del orador y productor, garantizando que el audio ingrese correctamente.
6. **Inyección Manual de Subtítulos**:
   - Barra de inyección instantánea para que los organizadores o el orador envíen avisos en vivo (*"Preguntas en 5 minutos"*, *"Speaker Q&A starting soon"*), traducidos automáticamente para la audiencia.
7. **Paneles Especializados para el Evento**:
   - **Audiencia (`/session/{id}`)**: Subtítulos legibles, selector de idioma en vivo (ES, EN, PT), tamaño de fuente ajustable y diseño responsivo.
   - **Pantalla de Escenario & Smart TV (`/stage/{id}`)**: Vista de alto contraste para auditorios y Smart TVs (50", 60", 75"+) con atajos de teclado (`F` fullscreen, `+/-` tamaño de fuente, `C` centrar) y código QR dinámico para acceso móvil.
   - **Micrófono del Orador (`/speaker/{id}`)**: Interfaz ligera para transmitir voz desde cualquier laptop o smartphone con visualizador de decibelios.
   - **Panel del Productor (`/producer`)**: Métricas de telemetría en tiempo real (`STT ms`, `TR ms`, `RTT ms`, `Total ms`), conteo de espectadores, gestión de salas y botón de **Borrado de Registro** sincronizado en vivo.
   - **Demostración de 2 Salas (`/demo`)**: Simulación interactiva lista para presentaciones con 2 escenarios en paralelo.
8. **Exportación de Subtítulos (.SRT, .VTT, .TXT)**:
   - Descarga instantánea de la transcripción completa de cualquier charla con marcas de tiempo sincronizadas relativas al inicio.
9. **Internacionalización Completa de la Interfaz**:
   - Selector de idioma global (🌐 ES / EN) integrado en la barra superior de todas las vistas.
10. **Seguridad y Producción**:
    - Protección por contraseña (`APP_PASSWORD`), cookies firmadas criptográficamente (`AUTH_SECRET`), protección CSRF/origen y Content-Security-Policy estricta.

---

## 🗺️ Mapa de URLs del Sistema

| Ruta | Rol / Vista | Descripción |
|---|---|---|
| `/` | **Hub de Audiencia** | Portada donde los asistentes eligen su sala (`Stage A`, `Stage B`, etc.) y cambian el idioma de la app. |
| `/session/{id}` | **Vista de Audiencia** | Subtítulos en vivo para los asistentes (ej. `/session/stage-a`), selector de idioma (ES, EN, PT), tamaño de letra ajustable y modo oscuro/claro. |
| `/stage/{id}` | **Pantalla de Escenario** | Vista limpia y de alto contraste para proyectores gigantes de escenario o integración con OBS Studio / vMix. |
| `/speaker/{id}` | **Micrófono del Orador** | Captura de audio directa desde el teléfono o laptop del orador con visualizador de audio en vivo e inyector manual. |
| `/producer` | **Panel del Productor** | Sala de control: telemetría de latencia en vivo (`STT`, `TR`, `RTT`, `Total`), contador de espectadores, gestión de escenarios, inyección manual y **Borrado de Registro** en vivo. |
| `/demo` | **Demo de 2 Salas** | Demostración interactiva en vivo con dos salas simultáneas transmitiendo en paralelo. |
| `/standalone` | **Laboratorio Clásico STT** | Banco de pruebas para comparar en tiempo real los 5 motores (Web Speech, Whisper local, Nemotron local, Deepgram, ElevenLabs). |
| `/health` | **Health Check** | Estado del servidor, disponibilidad de Gemini Live / fallback y conteo de salas/espectadores activos. |
| `/api/session/{id}/export/srt` | **Exportar SRT** | Descarga del archivo de subtítulos sincronizados en formato SubRip (.srt). |
| `/api/session/{id}/export/vtt` | **Exportar VTT** | Descarga de subtítulos en formato WebVTT (.vtt). |
| `/api/session/{id}/export/txt` | **Exportar TXT** | Descarga de la transcripción completa en texto plano (.txt). |

---

## 🚀 Guía de Inicio Rápido

### Requisitos Previos
- **Docker** y **Docker Compose** (Opción recomendada) ó **Python 3.10+**.
- Navegador moderno con soporte para WebGPU/Audio (Chrome, Edge, Brave o Firefox).

---

### Opción 1: Ejecutar con Docker Compose (Recomendado)

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/arsabot/nerdearla-live-transcribe.git
   cd nerdearla-live-transcribe
   ```

2. **Configurar el entorno**:
   ```bash
   cp .env.example .env
   ```
   *(Edita `.env` y añade tu `GEMINI_API_KEY` para activar la traducción simultánea por Gemini Live).*

3. **Iniciar el contenedor**:
   ```bash
   docker compose up --build -d
   ```

4. **Abrir en el navegador**:
   - Entra a **[http://localhost:3000](http://localhost:3000)**
   - Contraseña de acceso por defecto: `nerdearla2026`

---

### Opción 2: Ejecutar Localmente con Python

1. **Crear y activar un entorno virtual**:
   ```bash
   # En Linux / macOS:
   python3 -m venv .venv
   source .venv/bin/activate

   # En Windows (PowerShell):
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Configurar variables de entorno**:
   ```bash
   cp .env.example .env
   ```

4. **Ejecutar el servidor**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 3000 --reload
   ```

5. Accede a **[http://localhost:3000](http://localhost:3000)**.

---

## ⚙️ Configuración de Variables de Entorno (`.env`)

```ini
# --- Configuración del Servidor ---
PORT=3000                            # Puerto HTTP/WebSocket del servidor

# --- Autenticación y Seguridad ---
APP_PASSWORD=nerdearla2026           # Contraseña de acceso a la plataforma
AUTH_SECRET=nerdearla2026            # Clave para firmar cookies HMAC de sesión
AUTH_ENABLED=true                    # true / false para activar autenticación

# --- Proveedor de Traducción ---
TRANSLATION_PROVIDER=gemini_live     # 'gemini_live' (WebSocket streaming) o 'googletrans'
GEMINI_API_KEY=                      # Clave de Google AI Studio para Gemini Live
GEMINI_LIVE_MODEL=gemini-3.5-live-translate-preview

# --- Proveedores Opcionales de STT en la Nube ---
DEEPGRAM_API_KEY=                    # Token de Deepgram Nova-3 (opcional)
ELEVENLABS_API_KEY=                  # Token de ElevenLabs Scribe (opcional)

# --- Motores STT Habilitados ---
ENABLED_ENGINES=webspeech,whisper,gemini_live,nemotron,deepgram,elevenlabs
```

---

## 🧪 Ejecución de Pruebas Automatizadas

El proyecto cuenta con una suite completa de 66 pruebas unitarias e integrales que validan la autenticación, aislamiento multi-sala, WebSockets, Gemini Live y normalización de marcas:

```bash
# Ejecutar todas las pruebas
pytest -v

# Ejecutar con reporte de cobertura
pytest --cov=app --cov-report=term-missing
```

---
---

# 🇬🇧 English Version

## 📑 Table of Contents
- [🎯 What Problem Does It Solve?](#-what-problem-does-it-solve)
- [📖 What is Nerdearla Live?](#-what-is-nerdearla-live)
- [🏗️ System Architecture](#-system-architecture)
- [✨ Key Features](#-key-features)
- [🗺️ System Route Overview](#-system-route-overview)
- [🚀 Quick Start Guide](#-quick-start-guide)
  - [Option 1: Docker Compose (Recommended)](#option-1-run-with-docker-compose-recommended)
  - [Option 2: Local Python Virtualenv](#option-2-run-locally-with-python-virtual-environment)
- [⚙️ Environment Variables Reference (.env)](#-environment-variables-reference-env)
- [🧪 Running Automated Tests](#-running-automated-tests)

---

## 🎯 What Problem Does It Solve?

At large international tech conferences like **Nerdearla** (10,000+ attendees and dozens of concurrent talks), communication encounters major barriers:

1. **Language Barrier & Exclusion**:
   - International speakers present in **English or Portuguese**, while many local attendees prefer or need **Spanish** subtitles (and vice versa).
   - Hiring simultaneous human interpreter booths for 5, 10, or 30 concurrent stages is **prohibitively expensive** and logistically unfeasible.
2. **Real Accessibility for Hearing-Impaired Attendees**:
   - Attendees with **hearing loss or auditory challenges** are excluded without immediate, synchronized live captions.
   - In noisy convention halls or remote streams, keeping up with fast, complex technical jargon is difficult.
3. **Phonetic Distortion of Proper Names & Tech Terminology**:
   - Generic Speech-to-Text tools butcher tech jargon: turning *"Nerdearla"* into *"nerd de habla"* or *"merdearla"*, and mangling words like *Kubernetes, PRs, Pull Requests, DevOps, Open Source*.
4. **High Server Costs (GPU) & Audio Privacy**:
   - Transcribing continuous audio streams on cloud servers burns expensive GPU credits or sends sensitive audio off-premise.

**Nerdearla Live solves this completely**:
- **Zero server GPU cost**: Using on-device STT with Whisper and Nemotron over **WebGPU** directly in the speaker's browser (or Web Speech API), audio never leaves the device for transcription.
- **Sub-second simultaneous translation (< 700 ms)**: Bidirectional WebSocket connection to **Gemini Live API** with seamless automatic fallback to Google Translate.
- **Speech-Biasing & Normalization for "Nerdearla"**: Phonetic glossary that intercepts and corrects distortions in real time.
- **Isolated Multi-Stage Orchestration**: Instant broadcast of live subtitles to attendees' phones, auditorium screens, and streaming setups with strict room isolation.

---

## 📖 What is Nerdearla Live?

**Nerdearla Live** is a modular, self-hosted, open-source platform that turns live audio from multiple stages into **synchronized, highly accurate live transcripts and translations** (< 700 ms) for audience members, speakers, and stage projection screens.

Any speaker can simply open a link on their laptop or phone (`/speaker/stage-a`), unmute their mic, and broadcast multilingual subtitles to the audience (`/session/stage-a`) and to the auditorium screen (`/stage/stage-a`) with sub-second latency.

---

## 🏗️ System Architecture

```text
                                  NERDEARLA 2026
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
          STAGE A                    STAGE B                    STAGE N
      AI & Open Source           Cloud & DevOps               Security
             │                          │                          │
             ▼                          ▼                          ▼
    MICROPHONE + VISUALIZER    MICROPHONE + VISUALIZER    MANUAL INJECTION
   (/speaker/stage-a WebGPU)  (/speaker/stage-b Cloud)    (Staff Announcements)
             │                          │                          │
             ▼                          ▼                          ▼
      Speech-to-Text             Speech-to-Text             Manual Text
    • Whisper Local (WebGPU)   • Deepgram Nova-3          • Q&A Prompts
    • Nemotron-3.5 (WebGPU)    • ElevenLabs Scribe        • Sponsor Mentions
    • Web Speech API           • Web Speech API                    │
             │                          │                          │
             └──────────────────────────┼──────────────────────────┘
                                        ▼
                   REAL-TIME TRANSLATION PIPELINE + GLOSSARY
           ┌────────────────────────────────────────────────────────┐
           │ • Bidirectional WebSocket to Gemini Live API           │
           │ • Seamless & Automatic Fallback to Google Translate    │
           │ • Phonetic Normalization ("Nerdearla" + Tech Glossary) │
           └────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                         MULTI-ROOM SESSION MANAGER
                       (Strict Room Event Isolation)
                                        │
     ┌──────────────────────────────────┼──────────────────────────────────┐
     ▼                                  ▼                                  ▼
AUDIENCE VIEW                    STAGE DISPLAY                     CONTROL ROOM
(/session/{id})                  (/stage/{id})                     (/producer)
• Large readable subtitles       • High-contrast projector mode    • Real-time latency telemetry
• Live language switcher         • OBS Studio / vMix ready         • Active viewer counters
• Light / Dark mode              • Clutter-free UI                 • Live history reset
• .SRT / .VTT / .TXT export                                        • Subtitle injection
```

---

## ✨ Key Features

1. **Multi-Stage Orchestration**:
   - Run **2, 5, 10, or 30+ concurrent conference stages** (`Stage A`, `Stage B`, `Stage C`, etc.) with strict WebSocket event isolation.
   - Dynamic stage creation, status updates (active, paused, completed), and deletion in real time.
2. **5 Speech-to-Text Engines**:
   - **Whisper Local (ONNX / WebGPU / WASM)**: Runs directly in the speaker's browser via Transformers.js (zero server GPU cost & full audio privacy).
   - **Nemotron Local (ONNX Runtime Web)**: Client-side NVIDIA Nemotron-3.5-ASR streaming with WebGPU acceleration.
   - **Web Speech API**: In-browser zero-overhead speech recognition.
   - **Deepgram Nova-3**: Cloud streaming STT with high punctuation accuracy.
   - **ElevenLabs Scribe v2**: Real-time cloud streaming ASR.
3. **Real-Time Translation Pipeline with Gemini Live & Automatic Fallback**:
   - Direct bidirectional WebSocket connection to **Gemini Live API** (`gemini-3.5-live-translate-preview` / `gemini-2.0-flash-exp`).
   - Circuit breaker with seamless fallback to **Google Translate** during network interruptions or quota limits, with auto-recovery.
4. **"Nerdearla" Detection & Speech-Biasing**:
   - Automatic normalization of STT phonetic distortions (*"nerd de habla"*, *"nerd arla"*, *"nerdear la"*, *"merdearla"* -> `Nerdearla`).
   - Technical glossary mapping (*Kubernetes, Pull Request, DevOps, Open Source*).
5. **Live Audio Visualizer & Mic Monitor**:
   - Dynamic real-time Web Audio API frequency visualizer and volume meter in the speaker and producer terminals, preventing silent or misconfigured audio capture.
6. **Live Subtitle Injection**:
   - Instant manual injection bar for staff and speakers to broadcast urgent announcements or Q&A prompts with automatic translation for the audience.
7. **Role-Specific Views**:
   - **Audience (`/session/{id}`)**: High-contrast, large-font accessible subtitles with instant language switcher (ES, EN, PT).
   - **Stage Display & Smart TV (`/stage/{id}`)**: Presentation display optimized for large auditoriums and Smart TVs (50", 60", 75"+) with hotkeys (`F` fullscreen, `+/-` font size, `C` center) and floating audience QR code.
   - **Speaker Mic (`/speaker/{id}`)**: Lightweight broadcaster interface with live transcription feedback and audio meter.
   - **Producer Dashboard (`/producer`)**: Telemetry overview (`STT`, `TR`, `RTT`, `Total ms`), viewer counts, room creation, and live **Clear History** broadcast.
   - **2-Stage Live Demo (`/demo`)**: Interactive presentation mode showing 2 parallel tracks streaming simultaneously.
8. **Clean Subtitle Exports (.SRT, .VTT, .TXT)**:
   - Download synchronized subtitles with accurate relative timestamps.
9. **Full UI Internationalization**:
   - Global language toggle (🌐 ES / EN) in the top navigation of every page.
10. **Security & Production Readiness**:
    - Password protection (`APP_PASSWORD`), cryptographic HMAC session signing (`AUTH_SECRET`), CSRF/origin checks, and strict Content-Security-Policy.

---

## 🗺️ System Route Overview

| Path | View / Role | Purpose |
|---|---|---|
| `/` | **Audience Hub** | Landing page for attendees to select their stage and toggle UI language. |
| `/session/{id}` | **Audience View** | Live subtitles & language switcher for attendees (e.g. `/session/stage-a`). |
| `/stage/{id}` | **Stage Display** | High-contrast display for stage projectors & video switchers (OBS / vMix). |
| `/speaker/{id}` | **Speaker Microphone** | Mobile/desktop web mic for speakers with live audio visualizer and subtitle injection. |
| `/producer` | **Producer Dashboard** | Control room: real-time telemetry, stage creation, live subtitle injection, and history reset. |
| `/demo` | **2-Stage Live Demo** | Interactive simulation showcasing 2 parallel tracks streaming simultaneously. |
| `/standalone` | **Standalone Lab** | Original single-channel test bench for comparing all 5 STT engines. |
| `/health` | **Health Check** | Server status, Gemini Live / fallback availability, and active viewer metrics. |
| `/api/session/{id}/export/srt` | **Export SRT** | Download synchronized subtitles in SubRip format (.srt). |
| `/api/session/{id}/export/vtt` | **Export VTT** | Download subtitles in WebVTT format (.vtt). |
| `/api/session/{id}/export/txt` | **Export TXT** | Download clean full transcript in plain text (.txt). |

---

## 🚀 Quick Start Guide

### Prerequisites
- **Docker** & **Docker Compose** (Recommended) or **Python 3.10+**.
- Modern web browser with WebGPU & Audio support (Chrome, Edge, Brave, or Firefox).

---

### Option 1: Run with Docker Compose (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/arsabot/nerdearla-live-transcribe.git
   cd nerdearla-live-transcribe
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   ```

3. **Build & start the service**:
   ```bash
   docker compose up --build -d
   ```

4. **Open in browser**:
   - Navigate to **[http://localhost:3000](http://localhost:3000)**
   - Default login password: `nerdearla2026`

---

### Option 2: Run Locally with Python Virtual Environment

1. **Create and activate virtualenv**:
   ```bash
   # On Linux / macOS:
   python3 -m venv .venv
   source .venv/bin/activate

   # On Windows (PowerShell):
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   ```

4. **Launch the server**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 3000 --reload
   ```

5. Access **[http://localhost:3000](http://localhost:3000)**.

---

## ⚙️ Environment Variables Reference (`.env`)

```ini
# --- Server Configuration ---
PORT=3000                            # HTTP and WebSocket port

# --- Authentication & Security ---
APP_PASSWORD=nerdearla2026           # Master access password
AUTH_SECRET=nerdearla2026            # HMAC signing secret
AUTH_ENABLED=true                    # Enable/disable password gate

# --- Translation Engine ---
TRANSLATION_PROVIDER=gemini_live     # 'gemini_live' (bidirectional WebSocket) or 'googletrans'
GEMINI_API_KEY=                      # Google AI Studio API key
GEMINI_LIVE_MODEL=gemini-3.5-live-translate-preview

# --- Cloud STT Providers (Optional) ---
DEEPGRAM_API_KEY=                    # Deepgram Nova-3 API key
ELEVENLABS_API_KEY=                  # ElevenLabs Scribe v2 API key

# --- Engine Configuration ---
ENABLED_ENGINES=webspeech,whisper,gemini_live,nemotron,deepgram,elevenlabs
```

---

## 🧪 Running Automated Tests

```bash
# Run all unit and integration tests
pytest -v

# Run with coverage report
pytest --cov=app --cov-report=term-missing
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details. Built with ❤️ for **Nerdearla 2026**.

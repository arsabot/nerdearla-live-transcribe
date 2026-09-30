# Mapa Real de Arquitectura - Transcrively

## 1. Visión General del Flujo de Datos (Pipeline Real)

A continuación se detalla cómo fluyen los datos y eventos a través de los componentes reales del proyecto, de acuerdo a la inspección del código fuente (especialmente en `app/api/routes/websockets.py` y `app/templates/index.html`).

```text
[Frontend: Micrófono / Interfaz de Producción]
     │
     ├─ (Opciones STT) ──> WebSpeech API (Nativo, en navegador)
     ├─ (Opciones STT) ──> Transformers.js Whisper (WebGPU/WASM)
     ├─ (Opciones STT) ──> ONNX Nemotron-3.5 (WebGPU/WASM)
     └─ (Opciones STT) ──> Deepgram / ElevenLabs (WebSocket proxy)
     │
[Audio Processing & VAD / Segmentation]
     │
     └─> Se ejecuta íntegramente en el cliente (navegador del productor) a través
         de `app/templates/index.html`. El "chunker" (`SemanticChunker`) decide 
         cuándo cortar cadenas muy largas (interim) basándose en longitud y puntuación.
     │
[STT Output (Original Transcript)]
     │
     └─> El cliente (productor) emite un evento JSON `{"type": "interim" | "final"}`
         a través de un WebSocket hacia el Backend.
     │
[Backend: FastAPI (`/ws/session/{session_id}/producer`)]
     │
     └─> `session_producer_ws` recibe el texto original.
         Verifica si es necesario dividir el texto (`event_id-split`).
         │
         [Translation Pipeline]
         ├─> `TranslationProvider.translate_batch()`
         │    └─> Concurrencia mediante `asyncio.gather` para todos los idiomas objetivo.
         │    └─> Fallback chain: Gemini API -> Google Translate (`googletrans`).
         │
         [Composición del Evento de Transcripción]
         ├─> Se construye un objeto `TranscriptEvent` (con métricas de latencia, 
         │   texto original y traducciones completadas).
         │
         [Almacenamiento y Broadcast]
         └─> `SessionManager.broadcast_event(session_id, event)`
              ├─> Añade a `session.history` si es un evento "final".
              ├─> Desencadena un guardado asíncrono a disco (`trigger_save`).
              └─> Itera sobre un `Set` compartido (`session.viewers` + `session.producers`)
                  y emite el payload JSON concurrentemente mediante `asyncio.gather`.
     │
[WebSocket Clients (Viewers & OBS)]
     │
     ├─> `audience_session.html` (Múltiples idiomas, UI interactiva, selección de idioma local)
     ├─> `stage_display.html` (Salida fullscreen de alto contraste)
     └─> `obs_display.html` (Subtítulos limpios para broadcast, ignorando `interim`)
```

---

## 2. Componentes Clave e Identidad

### 2.1. Gestión de Estado (`SessionManager`)
- **Quién mantiene el estado:** La clase Singleton `SessionManager` (`app/session_manager.py`).
- **Cómo se almacena:** En memoria mediante un diccionario `_sessions`, protegido por un `asyncio.Lock()`.
- **Persistencia:** Se desencadena un hilo de respaldo a disco (`_save_to_storage_sync`) al archivo `data/sessions.json` en cada actualización importante.
- **Acumulación de datos:** Los eventos `final` se acumulan indefinidamente en la lista `session.history` en memoria y disco (riesgo de *memory leak* o degradación en sesiones muy largas).

### 2.2. Conexiones (WebSockets)
- **Quién abre conexiones:** 
  - El productor abre `ws://.../ws/session/{id}/producer`.
  - La audiencia y OBS abren `ws://.../ws/session/{id}/viewer`.
- **Quién cierra conexiones:** 
  - El cliente, al desconectarse o fallar la red.
  - El backend cerrará masivamente si la sesión se elimina (`delete_session`).
- **Problema de Concurrencia Detectado:** 
  - Las listas `session.viewers` y `session.producers` son colecciones `Set` nativas de Python.
  - En `broadcast_event`, se iteran estos conjuntos. Dado que la adición/eliminación ocurre en hilos/tareas asíncronas diferentes de FastAPI, hay una condición de carrera si el set cambia de tamaño durante la iteración.

### 2.3. Audio y Procesamiento (Frontend)
- **Quién inicia el proceso:** El botón "Start Microphone Broadcast" en `producer.html` o `index.html`.
- **Dónde se procesa:** Todo el STT (a excepción de Deepgram/ElevenLabs) ocurre directamente en el hilo principal del navegador o Web Workers (Transformers.js).
- **AudioContext Lifecycle:**
  - El frontend instancia un `new AudioContext()` cuando se selecciona un motor que requiere PCM (Whisper, Nemotron, Deepgram).
  - **Fuga (Leak) de Memoria detectada:** `audioContext.close()` *solo* se llama dentro de `stopDeepgramPipeline()`. Si un usuario alterna motores o reinicia continuamente la grabación local (WebSpeech/Whisper/Nemotron), se acumulan AudioContexts activos que nunca se cierran ni liberan el micrófono correctamente.

### 2.4. Tubería de Traducción (Backend)
- **Dónde se traduce:** En el backend (`session_producer_ws`), de forma bloqueante *lógica* pero no de red (se usa asincronismo para las APIs).
- **Proceso concurrente:** Las traducciones a múltiples idiomas (ej. ES, PT, FR) ocurren simultáneamente mediante `asyncio.gather`.
- **Fallback Activo:** El proveedor principal (Gemini) tiene un circuit breaker (`_is_circuit_open()`) basado en tiempo. Si falla por cuota (429) o timeout, dispara la protección por 45 segundos y recae instantáneamente en `googletrans`.
- **Acumulación de traducciones en progreso:** Las traducciones de eventos `interim` se sobreescriben lógicamente. Sin embargo, si un texto muy largo activa una división por parte del `SemanticChunker` (`event_id-split`), el frontend enviará eventos posteriores sin esperar a la traducción del primero.

# Transcrively — Technical Audit

## Executive Summary

La auditoría técnica integral del proyecto **Transcrively** revela que la plataforma cumple eficazmente su función de STT y traducción concurrente utilizando un enfoque innovador centrado en procesar el STT en el cliente (Browser/WebGPU) y delegar la traducción al backend. 

Sin embargo, para un entorno de producción crítica (como una conferencia en vivo), existen vulnerabilidades de alto riesgo relacionadas con el manejo de memoria en el cliente (fugas de AudioContext), concurrencia insegura en el manejo de colecciones del WebSocket en Python, y riesgos de agotamiento de memoria por historiales ilimitados.

A continuación, se detalla el análisis objetivo del código actual bajo la estricta directriz de **NO MODIFICAR** el proyecto.

---

## Tabla Resumen de Hallazgos

| ID | Severidad | Categoría | Problema | Archivo | Estado |
| -- | --------- | --------- | -------- | ------- | ------ |
| 1 | 🔴 CRITICAL | Concurrency | Iteración insegura de colecciones `Set` durante el broadcast de WebSockets que puede causar crashes (`Set changed size during iteration`). | `app/session_manager.py` | Detectado |
| 2 | 🔴 CRITICAL | Audio / Memory | Fuga (Leak) de instancias `AudioContext`. Solo se cierran en Deepgram, acumulando contextos (límite del navegador: 6) hasta romper el micrófono. | `app/templates/index.html` | Detectado |
| 3 | 🟠 HIGH | Dependency / API | `googletrans` se utiliza como fallback de Gemini. Esta librería hace scraping inestable y la IP del servidor será bloqueada bajo uso continuo en producción. | `app/translation_provider.py` | Detectado |
| 4 | 🟠 HIGH | State / Memory | `session.history` crece infinitamente guardando todos los eventos "final", pudiendo degradar la memoria del backend en sesiones de varias horas. | `app/session_manager.py` | Detectado |
| 5 | 🟡 MEDIUM | UI / UX | Desconexiones fantasma. Si la red se cae sin que el WebSocket reciba el evento `onclose`, la interfaz de audiencia no refleja la pérdida de conexión de inmediato. | `app/templates/audience_session.html` | Detectado |
| 6 | 🟡 MEDIUM | Logic / Translation | Los chunks `interim` extremadamente largos divididos forzosamente asumen latencia 0, superponiendo eventos si la traducción anterior falla. | `app/api/routes/websockets.py` | Detectado |
| 7 | 🟢 IMPROVEMENT | OBS | La vista de OBS carece de control automático de tamaño dinámico si el presentador no respira y la API no segmenta. (Mitinigado con `line-clamp` reciente). | `app/templates/obs_display.html` | Detectado |

---

## Architecture Findings

### [ID: 1] Condición de carrera en el Broadcast (RuntimeError: Set changed size)
- **Severidad:** 🔴 CRITICAL
- **Componente/función:** `SessionManager.broadcast_event` (`app/session_manager.py`)
- **Problema:** El backend itera sobre un conjunto (`set`) nativo de Python que combina `session.viewers` y `session.producers`.
- **Por qué ocurre:** Las listas de websockets se añaden o eliminan en corrutinas separadas. Si un usuario se conecta/desconecta exactamente en el momento en que se ejecuta `list(session.viewers) + list(session.producers)` o dentro de la iteración de broadcast, Python lanzará un `RuntimeError` y el evento se perderá.
- **Solución sugerida:** Cambiar a copias seguras de la lista o usar `.copy()` sobre el set antes de iterar bajo el `_lock`.

## Audio Findings

### [ID: 2] Fuga Masiva de `AudioContext`
- **Severidad:** 🔴 CRITICAL
- **Componente/función:** `stopRecording()` (`app/templates/index.html`)
- **Problema:** Los navegadores imponen un límite estricto (generalmente 6) de instancias de `AudioContext` concurrentes por pestaña.
- **Por qué ocurre:** `index.html` inicializa un `new AudioContext()` al iniciar Whisper o Nemotron. Sin embargo, en las funciones `stopWhisperPipeline()` o `stopNemotronPipeline()`, jamás se invoca `audioContext.close()`. Esto solo se hace en `stopDeepgramPipeline()`.
- **Impacto:** Si el productor presiona "Start" y "Stop" 6 veces, el micrófono y todo el procesamiento de audio fallará silenciosamente.

## STT Findings

- **Evaluación Positiva:** La delegación de WebSpeech y Transformers.js al cliente libera enormemente la CPU del backend. El VAD y la segmentación están adecuadamente integrados.

## Translation Findings

### [ID: 3] Uso inestable de `googletrans`
- **Severidad:** 🟠 HIGH
- **Componente/función:** `RobustTranslator` (`app/translation_provider.py`)
- **Problema:** En el caso de que Gemini dispare un *Circuit Breaker* (por ejemplo, por un HTTP 429), el sistema recae en `googletrans`. 
- **Impacto:** Esta biblioteca no es oficial y usa la API web pública de Google Translate. En un evento en vivo con tres idiomas simultáneos y múltiples mensajes `interim` por segundo, Google bloqueará la IP del servidor por abuso casi instantáneamente.

## WebSocket Findings

- Se verificaron múltiples conexiones asíncronas correctas.
- **Ausencia de Heartbeat activo bidireccional:** El sistema depende del `onclose` del navegador para iniciar la reconexión. En caídas de red "hard" (ej. pérdida de WiFi sin cierre de socket), el cliente y el servidor pueden quedar con sockets zombis durante varios minutos antes de detectar el timeout del SO.

## OBS Findings

- La integración OBS funciona y se alinea con estándares de broadcast gracias al manejo del DOM con `-webkit-line-clamp`. El reciente cambio donde se ignoran los eventos `interim` en `obs_display.html` mitiga exitosamente los parpadeos en pantalla.

## UI/UX Findings

### [ID: 5] Estados inmutables en caídas de red
- **Severidad:** 🟡 MEDIUM
- **Problema:** Si el WebSocket de un espectador pierde conectividad de forma opaca (sin evento `onclose`), la UI seguirá diciendo "Esperando transmisión" o mostrará el último subtítulo sin indicar un estado de reconexión.

## State Management Findings

### [ID: 4] Memory Leak Lógico en Session History
- **Severidad:** 🟠 HIGH
- **Componente/función:** `SessionManager.record_event`
- **Problema:** Cada evento `final` de STT se añade a la lista `session.history.append(event)`. En una conferencia de 8 horas ininterrumpidas, la RAM que consumen los objetos de Python puede dispararse, y cada `trigger_save()` demorará más tiempo escribiendo archivos gigantes en `data/sessions.json`.
- **Solución sugerida:** Mantener un `max_history_size` o rotar el almacenamiento.

## Concurrency Findings
- El sistema de traducción maneja las salidas en un `asyncio.gather()`. Esto es arquitectónicamente sólido: la traducción al portugués no retrasa la traducción al francés. (Escenarios A-G auditados satisfactoriamente).

## Security & Dependency Findings
- **Gemini API:** Como la integración de Gemini 3.5 Live Translate usa REST directo y WebSockets limpios, no hay vulnerabilidad de Key Leakage al frontend.
- **Riesgo:** El uso de bibliotecas de scraping para `googletrans` es deuda técnica grave. Se recomienda sustituir por `google-cloud-translate` oficial.

---

## Priority Roadmap

### AHORA (Antes de una demo en vivo)
1. **[ID: 2]** Reparar la fuga de `AudioContext` invocando `.close()` en todos los pipelines locales (`index.html`).
2. **[ID: 1]** Aplicar un `copy()` seguro a los `sets` de Websockets en el `SessionManager` para evitar caídas catastróficas durante la iteración en el broadcast.

### PRÓXIMO (Importante pero no bloqueante inmediato)
3. **[ID: 4]** Limitar el tamaño máximo de `session.history` o desvincular el guardado persistente del hilo principal (evitar IO Bound lock).

### DESPUÉS (Mejoras Técnicas)
4. **[ID: 3]** Desechar `googletrans` e implementar la API oficial de Google Cloud o permitir deshabilitar el fallback explícitamente para evitar bloqueos de IP en medio del evento.
5. **[ID: 5]** Implementar un mecanismo de *ping/pong* bidireccional cada 10 segundos en el WebSocket de los viewers para detectar caídas silenciosas de red.

---

## Pruebas Manuales Propuestas
Basado en los riesgos detectados, sugiero realizar las siguientes comprobaciones manuales antes del evento:

```text
[ ] Reconexión de red: Apagar y encender Wi-Fi como audiencia (verificar si la UI detecta la desconexión opaca).
[ ] Límite de AudioContext: En el productor, presionar Start -> Stop -> Start -> Stop repetidamente 8 veces y verificar si el micrófono sigue habilitándose.
[ ] Multilenguaje intensivo: Emitir English -> [Spanish, Portuguese, French] y hablar muy rápido para forzar un Rate Limit (429) de Gemini y verificar el comportamiento del Fallback de googletrans.
[ ] Iteración paralela: Abrir 15 pestañas de audiencia e intentar cerrarlas masivamente mientras el productor está transmitiendo a alta velocidad, para buscar el error de `Set changed size` en el backend.
[ ] Recuperación de OBS: Actualizar el servidor de backend (restart) y observar si el navegador incrustado en OBS se reconecta exitosamente después de 3 segundos sin asistencia manual.
```

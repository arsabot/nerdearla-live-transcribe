# Guía de Configuración para OBS Studio

Esta guía explica cómo capturar los subtítulos generados en vivo y enviarlos a OBS Studio para añadirlos directamente a tus transmisiones o grabaciones en vivo.

La aplicación incluye una vista especial diseñada específicamente para OBS. A diferencia de las vistas normales (donde puedes ver la barra superior o las opciones de idioma), la vista de OBS tiene un **fondo completamente transparente** y mantiene los subtítulos anclados en la parte inferior para que se superpongan de manera limpia sobre tu cámara.

## Paso a Paso en OBS

1. **Abre OBS Studio.**
2. En la sección de **Fuentes (Sources)** (abajo al centro), haz clic en el botón `+`.
3. Selecciona **Navegador (Browser)**.
4. Dale un nombre a la fuente, por ejemplo: `Subtítulos Live Transcribe`.
5. Se abrirá la ventana de propiedades. Configura los siguientes campos:
   - **URL**: Ingresa la ruta especial hacia tu sala. En tu caso, para la sala `stage-b`:
     `http://localhost:3000/obs/stage-b`
     
     **IMPORTANTE**: 
     - Como la aplicación requiere contraseña, necesitas pasar la contraseña añadiendo `?pwd=TU_CONTRASEÑA`.
     - Si la sala transmite en varios idiomas y tú solo quieres mostrar uno en OBS, usa `&lang=es` (o `en`).
     
     Por ejemplo, para acceder con tu contraseña `admin` y forzar solo subtítulos en español:
     `http://localhost:3000/obs/stage-b?pwd=admin&lang=es`
   - **Ancho (Width)**: `1920` (o la resolución base de tu transmisión, ej. 1280).
   - **Alto (Height)**: `1080` (o la resolución base, ej. 720).
   - **CSS personalizado (Custom CSS)**: Puedes borrar el CSS que viene por defecto, ya que la aplicación se encarga de forzar el fondo transparente. Si quieres asegurarte, déjalo así:
     ```css
     body { background-color: rgba(0, 0, 0, 0); margin: 0px auto; overflow: hidden; }
     ```
6. Haz clic en **Aceptar (OK)**.

## Ajustando la vista

Verás que los subtítulos se posicionarán en la parte inferior del lienzo de OBS y el fondo será transparente. 

Si deseas cambiar el tamaño de los subtítulos, los colores o hacer los textos más gruesos, puedes editar directamente las propiedades CSS en el archivo `app/templates/obs_display.html`. Algunas variables útiles que puedes modificar en ese archivo son:

- `--font-size: 3.6rem;` (Tamaño del texto, ajústalo según necesites).
- `--text-main: #fbb212;` (El color principal de los subtítulos).

¡Con esto ya tienes tus subtítulos traduciéndose en tiempo real directamente en tu overlay de OBS!

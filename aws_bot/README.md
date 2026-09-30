# AWS Serverless Summary Bot

Este directorio contiene el código para desplegar un bot serverless en AWS usando **AWS Lambda** y **Amazon Bedrock**. Esto te permite generar resúmenes del chat usando tu tarjeta de regalo de AWS, evitando costos en otras plataformas.

## Requisitos
1. Una cuenta de AWS.
2. Acceso a **Amazon Bedrock** (debes habilitar el acceso al modelo `Claude 3 Haiku` desde la consola de Bedrock en la región que vayas a usar, ej: `us-east-1`).
3. Crear una **AWS Lambda Function**.

## Pasos para Desplegar

1. **Crear la Lambda Function**:
   - Ve a la consola de AWS Lambda.
   - Crea una nueva función desde cero.
   - Nombre: `LiveTranscribeSummaryBot`
   - Runtime: `Python 3.12` (o superior).
   - Arquitectura: `x86_64` o `arm64`.

2. **Añadir el Código**:
   - Copia el contenido de `lambda_function.py` y pégalo en el editor de código integrado de Lambda.
   - Haz clic en **Deploy** (Desplegar).

3. **Configurar Permisos**:
   - Ve a la pestaña **Configuración (Configuration)** > **Permisos (Permissions)** de tu Lambda.
   - Haz clic en el nombre del **Rol de ejecución (Execution role)**.
   - Añade una política en línea (Inline Policy) para darle acceso a Bedrock:
     ```json
     {
         "Version": "2012-10-17",
         "Statement": [
             {
                 "Effect": "Allow",
                 "Action": "bedrock:InvokeModel",
                 "Resource": "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-haiku-20240307-v1:0"
             }
         ]
     }
     ```

4. **Habilitar Function URL (URL de la función)**:
   - En la consola de Lambda, ve a **Configuración** > **URL de la función**.
   - Crea una nueva Function URL.
   - Tipo de autenticación: **NONE** (para simplificar, o configúralo con IAM si planeas hacerlo privado y firmar las peticiones).
   - Guarda y copia la URL generada (ej: `https://abcd123.lambda-url.us-east-1.on.aws/`).

5. **Configurar el Proyecto**:
   - Abre el archivo `.env` en tu proyecto de Nerdearla Live Transcribe.
   - Añade la URL que copiaste:
     ```env
     AWS_SUMMARY_BOT_URL=https://abcd123.lambda-url.us-east-1.on.aws/
     ```

¡Listo! El botón de Resumen AI de la audiencia ahora enviará el historial a tu Lambda en AWS, que consumirá tus créditos de la tarjeta de regalo mediante Amazon Bedrock (Claude 3 Haiku).

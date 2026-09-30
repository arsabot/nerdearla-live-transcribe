import json
import boto3

# Initialize the Bedrock client
bedrock = boto3.client('bedrock-runtime')

def lambda_handler(event, context):
    try:
        # If the request comes from API Gateway / Function URL, the body is a JSON string
        if "body" in event:
            body = json.loads(event["body"])
        else:
            body = event
            
        text_to_summarize = body.get("text", "")
        language = body.get("language", "es")
        
        if not text_to_summarize:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No text provided to summarize."})
            }
            
        # We use Claude 3 Haiku for fast, cheap summaries
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        
        prompt = f"Resume brevemente (en 2-3 oraciones) el siguiente contenido. Escribe el resumen en el idioma en que está el texto:\n\n{text_to_summarize}"
        
        # Bedrock Claude Messages API format
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        response = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json"
        )
        
        response_body = json.loads(response.get("body").read())
        summary_text = response_body.get("content", [{}])[0].get("text", "No summary generated.")
        
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({"summary": summary_text})
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Failed to generate summary.", "details": str(e)})
        }

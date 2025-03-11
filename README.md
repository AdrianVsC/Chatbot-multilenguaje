# Chatbot Prueba Técnica Nolatech

Chatbot Adrian es un asistente conversacional basado en Azure OpenAI, langchain y Streamlit. Puede responder en múltiples idiomas, recordar conversaciones previas y recomendar canciones según el sentimiento del usuario.
Este proyecto se apoya en Langchain, tanto en sus herramientas como en su capacidad para encadenar funciones con el fin de conectar con la API de Youtube y recordar conversaciones que se guardaban tanto en la memoria RAM, como en la base SQLite3

## 🚀 Instalación

1. **Clona el repositorio** o descarga los archivos en tu máquina.
2. **Instala las dependencias** ejecutando:

   ```bash
   pip install -r requirements.txt

   ```

3. **Ejecuta el Script** ejecutando:

   ```bash
   streamlit run app.py
   ```

   **Nota:** Se necesita configurar el archivo .env con las keys y endpoints para que el modelo funcione correctamente

## Video de demostración

![Ver Video](src/Prueba%20de%20funcionamiento.gif)

## En el video se puede observar lo siguiente:

1. Cómo las conversaciones se separan y se almacenan por sessión y el lenguaje se separa por usuario
2. Cómo funciona la recuperación de memoria, ya que el chat puede recordar cuál es mi nombre
3. Cómo se conecta con la api de YT y obtiene el link de un video motivacional si reconoce que el sentimiento del prompt es negativo
   **Nota:** Se debe comunicar en inglés porque el analizador de sentimiento solo reconoce en inglés y si se escribe en otro idioma lo reconocerá como sentimiento neutro

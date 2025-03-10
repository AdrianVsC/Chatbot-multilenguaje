import os
import sqlite3
import uuid
import datetime
import re
from contextlib import contextmanager
import asyncio  

# Langchain + Azure OpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import AzureChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ChatMessageHistory

# YouTube Search Tool
from langchain_community.tools import YouTubeSearchTool
import yt_dlp

# Detección de Lenguaje y Sentimiento
from langdetect import detect
import langcodes
from textblob import TextBlob  

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv()

'''
Este codigo es un ejemplo de como se puede implementar un chatbot multilingue con Azure OpenAI, langchain y SQLite.
En este caso, intenté integrar el chatbot con una api externa, en este caso Youtube para buscar videos según el sentimiento del usuario.
'''

# Path de la base de datos
DB_PATH = "db/chatbot_prueba.db"

# Variables de entorno
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")


# Función para inicializar la base de datos
def setup_database():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS usuarios (
            user_id TEXT PRIMARY KEY,
            ultimo_acceso TEXT,
            preferencias TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS historial_mensajes (
            message_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            idioma TEXT NOT NULL, 
            sentimiento TEXT,
            timestamp TEXT,
            FOREIGN KEY (user_id) REFERENCES usuarios(user_id)
        )
        ''')

        conn.commit()


# Función para manejar conexiones a la base de datos
@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# Clase del Chatbot
class MultilingualAzureChatbot:
    ram_storage = {}  # Memoria temporal

    def __init__(self):
        self.llm = AzureChatOpenAI(
            openai_api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            temperature=0.7,
            max_tokens=1000,
        )

        # Inicializar YouTube Search Tool
        self.youtube_tool = YouTubeSearchTool()

        self.prompt = ChatPromptTemplate.from_messages([
            ('system', """
            ***REGLAS DE CONVERSACIÓN***
             1.Recomienda canciones alegres según el sentimiento del usuario: {sentimiento}.
             2.La recomendación de la canción incluirá el nombre de la canción y un enlace de YouTube: {cancion_recomendada}.
             3.Responde en el idioma preferido del usuario: {idioma}.
            """),
            MessagesPlaceholder(variable_name='history'),
            ('human', '{input}'),
        ])
    
        self.runnable = self.prompt | self.llm    

    def detectar_idioma(self, texto: str) -> str:
        """Detecta el idioma del texto y devuelve su nombre completo."""
        try:
            codigo_idioma = detect(texto)  # Detecta el idioma pero en formato es, en, it, etc.
            nombre_idioma = langcodes.Language.make(language=codigo_idioma).display_name("es")  # Convierte a nombre completo
            return nombre_idioma
        except:
            return "English" 
    
    def idioma_preferido_usuario(self, usuario_id: str) -> str:
        """Obtiene el idioma preferido del usuario"""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT idioma, COUNT(idioma) as conteo FROM historial_mensajes WHERE user_id = ? AND role = 'human' GROUP BY idioma ORDER BY conteo DESC", (usuario_id,))
            usuario = cursor.fetchone()
            if usuario:
                return usuario[0]
            else:
                return "English"

    def analizar_sentimiento(self, texto: str) -> str:
        """Analiza el sentimiento del texto usando TextBlob"""
        try:
            polaridad = TextBlob(texto).sentiment.polarity
            return "positivo" if polaridad > 0.1 else "negativo" if polaridad < -0.1 else "neutral"
        except:
            return "neutral"

    def obtener_titulo_video(self, enlaces: str) -> str:
        """Obtiene el título de un video de YouTube de manera confiable con `yt-dlp` sin mostrar advertencias."""
        try:
            ydl_opts = {
                "quiet": True,  
                "no_warnings": True,  
                "postprocessors": [],  
            }
        
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(enlaces, download=False)  
                return str(info["title"]) if "title" in info else "Título no encontrado"
        
        except Exception as e:
            print(f"Error al obtener título: {e}")
            return "Error al obtener título"



    def buscar_videos_por_sentimiento(self, sentimiento: str, idioma: str) -> str:
        """Busca videos de YouTube según el sentimiento e idioma del usuario y obtiene el título de la primera URL encontrada"""
        consulta = ""
        if sentimiento == "negativo":
            consulta = "canciones motivacionales para animarse" if "español" in idioma.lower() else "uplifting videos to cheer up"
        elif sentimiento == "positivo":
            consulta = "canciones divertidas para estar feliz" if "español" in idioma.lower() else "funny videos to stay happy"

        try:
            print("Buscando videos en YouTube...")
            resultados = self.youtube_tool.run(consulta)  # Devuelve un string con una URL

            print(f"Resultado obtenido: {resultados}")

            # 🔹 Extraer la primera URL del string usando una expresión regular
            match = re.search(r'https?://[^\s]+', resultados)

            if not match:
                raise ValueError("No se encontró una URL válida en el resultado.")

            enlace = match.group(0)  # 🔹 Extrae la primera URL encontrada
            print(f"Enlace extraído: {enlace}")

            # 🔹 Obtener título del video
            titulo = self.obtener_titulo_video(enlace)
            print(f"Título del video: {titulo}")

            return f"- {titulo}: {enlace}"
        except Exception as e:
            print(f"Error al buscar videos: {e}")
            return "Error al obtener videos."


    @classmethod
    def get_session_history(cls, session_id: str) -> ChatMessageHistory:
        """Recupera el historial de la sesión desde la base de datos y lo actualiza en la memoria"""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT role, content FROM historial_mensajes 
                WHERE session_id = ? 
                ORDER BY timestamp ASC
                LIMIT 4
            """, (session_id,))
            messages = cursor.fetchall()

            history = ChatMessageHistory()
            for row in messages:
                if row["role"] == "human":
                    history.add_message(HumanMessage(content=row["content"]))
                elif row["role"] == "ai":
                    history.add_message(AIMessage(content=row["content"]))
                elif row["role"] == "system":
                    history.add_message(SystemMessage(content=row["content"]))

            # ACTUALIZA memoria en 'ram_storage' con la versión más reciente
            cls.ram_storage[session_id] = history

            return history
            
    def guardar_mensaje_historial(self, conn, usuario_id, session_id, role, content):
        """Guarda un mensaje en el historial"""
        idioma = self.detectar_idioma(content)
        sentimiento = self.analizar_sentimiento(content)
        cursor = conn.cursor()
        id_mensaje = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        cursor.execute('''
        INSERT INTO historial_mensajes (message_id, session_id, user_id, role, content, idioma, sentimiento, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (id_mensaje, session_id, usuario_id, role, content, idioma, sentimiento, timestamp))
        
        conn.commit()


    async def generar_respuesta(self, usuario_id: str, session_id: str, mensaje: str):
        """Genera una respuesta ajustada según el sentimiento y el idioma preferido de la sesion"""
        idioma_preferido = self.idioma_preferido_usuario(usuario_id)
        sentimiento = self.analizar_sentimiento(mensaje)

        with get_db() as conn:
            history = self.get_session_history(session_id)

            # Guardar mensaje del usuario en el historial
            self.guardar_mensaje_historial(conn, usuario_id, session_id, "human", mensaje)

            # Si el historial está vacío, agregar un mensaje de contexto
            if not history.messages:
                history.add_message(SystemMessage(content="Eres un asistente útil y amigable. Responde de forma adecuada según el contexto."))

            # Ajustar la respuesta según el sentimiento
            if sentimiento == "negativo":
                history.add_message(SystemMessage(content="El usuario parece estar molesto o frustrado. Sé empático y considerado."))
            elif sentimiento == "positivo":
                history.add_message(SystemMessage(content="El usuario está animado. Responde con entusiasmo."))

            # Generar respuesta con contexto
            with_message_history = RunnableWithMessageHistory(
                self.runnable,
                self.get_session_history,
                input_messages_key='input',
                history_messages_key='history'
                )
            
            res = with_message_history.invoke(
            {'idioma':idioma_preferido ,'sentimiento':sentimiento,'input':mensaje},
            config={'configurable':{'session_id':session_id}})

            # Obtener la respuesta limpia
            respuesta_final = str(res.content)

            # Guardar respuesta del chatbot en el historial
            self.guardar_mensaje_historial(conn, usuario_id, session_id, "ai", respuesta_final)
            print('Se guardó con exito')
            print(f'historia recuperada: {history}')

            return respuesta_final



# Prueba del Chatbot
async def main():
    setup_database()
    bot = MultilingualAzureChatbot()

    print("🔄Chatbot iniciado. Escribe 'exit' para salir.")

    while True:
        prompt = input("👤 Usuario: ")  # Recibe el input del usuario
        if prompt.lower() == "exit":
            print("Saliendo del chatbot...")
            break  # Cierra el programa

        respuesta = await bot.generar_respuesta("Angelo", "session_010", prompt)
        print(f"Chatbot: {respuesta}\n")

if __name__ == "__main__":
    asyncio.run(main())  #Ejecutamos el loop de asyncio correctamente

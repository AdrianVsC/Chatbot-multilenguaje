import streamlit as st
import sqlite3
import asyncio
from llm import MultilingualAzureChatbot, setup_database

# Inicializar la base de datos
setup_database()

'''
En este Script se configura la interfaz para el chatbot usando streamlit y se conecta con la base de datos para almacenar el historial de mensajes.
'''


# Crear una instancia del chatbot
bot = MultilingualAzureChatbot()

# Configurar la interfaz
st.title("Chatbot Multilingual con Azure OpenAI")

# Identificador del usuario y sesión
usuario_id = st.text_input("🆔 Ingresa tu nombre de usuario:", "")
session_id = st.text_input("🔄 ID de sesión:", "")

# Mostrar historial de conversación con `st.chat_message()`
st.subheader("Conversación")

with sqlite3.connect("db/chatbot.db") as conn:
    cursor = conn.cursor()
    cursor.execute("""
        SELECT role, content 
        FROM historial_mensajes 
        WHERE session_id = ? 
        ORDER BY timestamp ASC 
        LIMIT 4
    """, (session_id,))
    
    mensajes = cursor.fetchall()

    for role, content in mensajes:
        with st.chat_message("user" if role == "human" else "assistant"):
            st.markdown(content)

# Cuadro de texto para que el usuario escriba su mensaje
mensaje = st.chat_input("Escribe tu mensaje:")

# Botón para enviar el mensaje
if mensaje:
    if mensaje:
        with st.spinner("Procesando..."):
            # Ejecutar la respuesta en modo asíncrono
            respuesta = asyncio.run(bot.generar_respuesta(usuario_id, session_id, mensaje))

            # Mostrar el mensaje del usuario
            with st.chat_message("user"):
                st.markdown(mensaje)

            # Mostrar la respuesta del chatbot
            with st.chat_message("assistant"):
                st.markdown(respuesta)
    else:
        st.warning("Escribe un mensaje antes de enviar.")
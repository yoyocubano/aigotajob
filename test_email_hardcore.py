import os
import json
import time
import sys
from datetime import datetime
from aigotajob_bot import AiGotAJobBot

def test_email_system():
    print("🛰️ INICIANDO PRUEBA DE FUEGO: SISTEMA DE NOTIFICACIÓN POR EMAIL...", flush=True)
    bot = AiGotAJobBot()
    
    subject = "PRUEVA DE FUEGO ADRENALINA"
    message = "Este es un mensaje de prueba para confirmar el blindaje total de la comunicación WeLux. Si recibes esto, el canal es OFICIALMENTE SEGURO."
    
    print(f"📧 Intentando enviar email a yucolaguilar@gmail.com...", flush=True)
    bot.notify_user(subject, message)
    
    print("\n✅ PROCESO DE PRUEBA TERMINADO.", flush=True)
    print("⚠️ POR FAVOR, COMANDANTE: CONFIRME SI RECIBIÓ EL EMAIL PARA CONTINUAR.", flush=True)

if __name__ == "__main__":
    test_email_system()

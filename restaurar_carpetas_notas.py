import subprocess
import sys
import re

def run_applescript(script):
    try:
        process = subprocess.Popen(['osascript', '-e', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=120)
        return stdout.decode('utf-8').strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

def restore_structure():
    print("🛰️ RESTAURANDO ESTRUCTURA DE CARPETAS INTELIGENTE...")
    sys.stdout.flush()

    CATEGORIES = {
        "⚡️ Ingeniería Eléctrica": ["#elec", "#⚡️-Ingeniería-Eléctrica", "electricidad", "knx", "pif", "volt", "maitrise", "estudios"],
        "👥 Gestión Familiar": ["#👥-Gestión-Familia", "mama", "papa", "familia", "reunificacion", "visa", "tramites"],
        "💰 Ecosistema Financiero": ["#💰-Ecosistema-Financiero", "crypto", "trading", "banco", "finanzas", "cuentas", "inversion"],
        "🦾 Laboratorio IA & Tech": ["#🦾-Laboratorio-IA-&-Tech", "ia", "ai", "swarm", "antigravity", "code", "software", "appwrite", "gpt"],
        "🩺 Salud": ["#🩺-Centro-de-Salud", "salud", "médical", "dieta", "ejercicio", "dentista", "gym"],
        "📖 Bitácora de Vida": ["#Bitacora_Sync", "bitacora", "log", "misión", "objetivo", "diario"],
        "🛠️ Operaciones LeadGen": ["lead", "hoplr", "facebook", "editus", "comercial", "negocio", "clientes"],
        "📂 Archivo General": []
    }

    # 1. Asegurar que las carpetas existen
    print("   🏗️ Creando carpetas operativas...")
    for cat in CATEGORIES.keys():
        run_applescript(f'tell application "Notes" to if not (exists folder "{cat}") then make new folder with properties {{name:"{cat}"}}')

    # 2. Obtener todas las notas de la carpeta principal "Notes"
    print("   📊 Analizando notas en la carpeta raíz para redistribución...")
    ids_raw = run_applescript('tell application "Notes" to get id of every note in folder "Notes"')
    if not ids_raw or "ERROR" in ids_raw:
        print("   ✅ No hay más notas en el Root o error en lectura.")
        return

    note_ids = [i.strip() for i in ids_raw.split(",")]
    print(f"   🔎 Encontradas {len(note_ids)} notas para clasificar.")
    
    for nid in note_ids:
        res = run_applescript(f'tell application "Notes" to return name of note id "{nid}" & "|||" & body of note id "{nid}"')
        if "|||" not in res: continue
        
        name, body = res.split("|||", 1)
        content_lower = (name + " " + body).lower()
        
        target_folder = "📂 Archivo General"
        for folder, triggers in CATEGORIES.items():
            if any(t.lower() in content_lower for t in triggers):
                target_folder = folder
                break
        
        # Mover a la carpeta correspondiente
        move_script = f'''
        tell application "Notes"
            try
                set n to note id "{nid}"
                set f to folder "{target_folder}"
                move n to f
            end try
        end tell
        '''
        run_applescript(move_script)

    print("✨ RESTAURACIÓN COMPLETADA. Las notas han sido organizadas por contenido y etiquetas.")
    sys.stdout.flush()

if __name__ == "__main__":
    restore_structure()

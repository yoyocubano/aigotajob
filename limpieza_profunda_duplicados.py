import subprocess
import difflib
import time
import sys

def run_applescript(script):
    try:
        process = subprocess.Popen(['osascript', '-e', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=120)
        return stdout.decode('utf-8').strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

def cleaning_operation(folder_name):
    print(f"\n🚀 OPERACIÓN LIMPIEZA INICIADA: {folder_name}")
    sys.stdout.flush()
    
    # 1. Obtener IDs y Títulos
    # AppleScript to get id and name of every note in folder
    script = f'tell application "Notes" to get {{id, name}} of every note of folder "{folder_name}"'
    res = run_applescript(script)
    
    if "ERROR" in res or not res:
        print(f"❌ Error al acceder a {folder_name}")
        return

    # res format is usually: id1, id2, name1, name2...
    # Split the result into IDs and Names
    parts = res.split(", ")
    mid = len(parts) // 2
    ids = parts[:mid]
    names = parts[mid:]
    
    total = len(ids)
    print(f"📊 Detectadas {total} notas. Analizando contenidos...")
    
    processed = [] # list of {id, name, body}
    to_delete = []

    for i in range(total):
        nid = ids[i].strip()
        nname = names[i].strip()
        
        # Get body for deep comparison
        body = run_applescript(f'tell application "Notes" to get body of note id "{nid}"')
        
        is_dupe = False
        for p in processed:
            # Requisito 1: Título idéntico
            if nname.lower() == p['name'].lower():
                # Requisito 2: Similitud de contenido > 90%
                ratio = difflib.SequenceMatcher(None, body, p['body']).ratio()
                if ratio > 0.9:
                    is_dupe = True
                    to_delete.append(nid)
                    break
        
        if not is_dupe:
            processed.append({"id": nid, "name": nname, "body": body})
        
        if (i+1) % 10 == 0:
            print(f"   ⏳ {i+1}/{total} analizadas...")
            sys.stdout.flush()

    # 2. Ejecutar Borrado Controlado (Velocidad Humana para no saturar iCloud)
    if to_delete:
        print(f"🗑️ ENCONTRADOS {len(to_delete)} DUPLICADOS. Iniciando borrado seguro...")
        for i, nid in enumerate(to_delete):
            run_applescript(f'tell application "Notes" to delete note id "{nid}"')
            print(f"   [{i+1}/{len(to_delete)}] Duplicado eliminado. Esperando sincronización...")
            time.sleep(3) # Pausa de 3 segundos entre borrados para dejar respirar a iCloud
    else:
        print("✅ No se detectaron duplicados críticos en esta carpeta.")

if __name__ == "__main__":
    cleaning_operation("⚡️ Ingeniería Eléctrica")
    cleaning_operation("🦾 Laboratorio IA & Tech")
    print("\n✨ LIMPIEZA PROFUNDA COMPLETADA. iCloud debería estar más ligero ahora.")

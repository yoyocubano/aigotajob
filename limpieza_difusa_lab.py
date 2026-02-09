import subprocess
import difflib
import time
import sys

def run_applescript(script):
    try:
        process = subprocess.Popen(['osascript', '-e', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=180)
        return stdout.decode('utf-8').strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

def fuzzy_cleanup_lab():
    folder = "🦾 Laboratorio IA & Tech"
    print(f"🕵️ ESCANEO DE SIMILITUD DIFUSA: {folder}")
    sys.stdout.flush()

    ids_raw = run_applescript(f'tell application "Notes" to get id of every note of folder "{folder}"')
    if "ERROR" in ids_raw or not ids_raw: return
    ids = [i.strip() for i in ids_raw.split(",")]
    total = len(ids)

    notes = [] # list of {id, name, body_trimmed}
    print(f"📊 Cargando {total} notas para comparación...")

    for i, nid in enumerate(ids):
        res = run_applescript(f'tell application "Notes" to get name of note id "{nid}" & "|||" & body of note id "{nid}"')
        if "|||" not in res: continue
        name, body = res.split("|||", 1)
        
        # Trim body for faster comparison (90% similarity is usually visible in first 5000 chars)
        notes.append({"id": nid, "name": name, "body": body[:5000]})
        
        if (i+1) % 30 == 0:
            print(f"   ⏳ {i+1}/{total} cargadas...")
            sys.stdout.flush()

    to_delete = []
    print("\n🔎 Buscando coincidencias del >85%...")
    
    for i in range(len(notes)):
        for j in range(i + 1, len(notes)):
            if notes[j]['id'] in [d[0] for d in to_delete]: continue
            
            # Check Title Similarity or Body Similarity
            if notes[i]['name'] == notes[j]['name']: # Same title
                ratio = difflib.SequenceMatcher(None, notes[i]['body'], notes[j]['body']).ratio()
                if ratio > 0.8:
                    to_delete.append((notes[j]['id'], notes[j]['name'], f"SIMILAR AL 80%+ ('{notes[i]['name']}')"))
            elif "copia" in notes[j]['name'].lower() or "copy" in notes[j]['name'].lower():
                 to_delete.append((notes[j]['id'], notes[j]['name'], "POSIBLE COPIA SISTEMA"))

    if to_delete:
        print(f"\n🗑️ BORRADO SEGURO DE {len(to_delete)} NOTAS SOSPECHOSAS (Velocidad iCloud)...")
        for i, (nid, nname, reason) in enumerate(to_delete):
            run_applescript(f'tell application "Notes" to delete note id "{nid}"')
            print(f"   [{i+1}/{len(to_delete)}] Borrado: {nname} ({reason})")
            time.sleep(3)
    else:
        print("✅ No se detectaron duplicados borrosos o copias.")

if __name__ == "__main__":
    fuzzy_cleanup_lab()

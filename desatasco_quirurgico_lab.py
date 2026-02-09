import subprocess
import time
import sys

def run_applescript(script):
    try:
        process = subprocess.Popen(['osascript', '-e', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=120)
        return stdout.decode('utf-8').strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

def surgical_desatasco_lab():
    folder = "🦾 Laboratorio IA & Tech"
    print(f"🕵️ ANALIZANDO PUNTOS DE ATASCO EN: {folder}")
    sys.stdout.flush()

    # 1. Obtener IDs (los IDs no suelen tener comas, son más seguros de parsear)
    ids_raw = run_applescript(f'tell application "Notes" to get id of every note of folder "{folder}"')
    if "ERROR" in ids_raw or not ids_raw:
        print(f"❌ Error al obtener IDs: {ids_raw}")
        return

    ids = [i.strip() for i in ids_raw.split(",")]
    total = len(ids)
    print(f"📊 {total} notas detectadas. Escaneando pesos y contenidos...")

    seen_content_hashes = {}
    to_delete = []

    for i, nid in enumerate(ids):
        # Fetch name and body for each note
        metadata = run_applescript(f'''
            tell application "Notes"
                set n to note id "{nid}"
                return name of n & "|||" & body of n
            end tell
        ''')
        
        if "|||" not in metadata: continue
        
        name, body = metadata.split("|||", 1)
        size = len(body)
        
        # Identify junk: Exact body duplicates, empty fragments, or massive logs
        if body in seen_content_hashes:
            to_delete.append((nid, name, "DUPLICADO EXACTO"))
        elif size < 50:
            to_delete.append((nid, name, "FRAGMENTO VACÍO"))
        elif size > 50000:
            # Massive notes often block sync. I'll flag but ONLY delete if they look like logs/duplicates
            if "log" in name.lower() or "output" in name.lower():
                to_delete.append((nid, name, f"LOG MASIVO ({size} chars)"))
            else:
                print(f"   ⚠️ NOTA MUY PESADA: '{name}' ({size} chars). Manteniéndola por seguridad.")
        else:
            seen_content_hashes[body] = nid

        if (i+1) % 15 == 0:
            print(f"   ⏳ {i+1}/{total} analizadas...")
            sys.stdout.flush()

    # 2. BORRADO SEGURO
    if to_delete:
        print(f"\n🗑️ PROCEDIENDO AL BORRADO QUIRÚRGICO DE {len(to_delete)} NOTAS TRABADAS...")
        for i, (nid, nname, reason) in enumerate(to_delete):
            run_applescript(f'tell application "Notes" to delete note id "{nid}"')
            print(f"   [{i+1}/{len(to_delete)}] Borrado: {nname} ({reason})")
            time.sleep(2) # Respiro para el motor de iCloud
    else:
        print("\n✅ No se encontraron duplicados exactos ni basura evidente.")

    print("\n✨ OPERACIÓN DE DESATASCO FINALIZADA.")

if __name__ == "__main__":
    surgical_desatasco_lab()

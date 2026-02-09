import subprocess
import difflib
import sys

def run_applescript(script):
    try:
        process = subprocess.Popen(['osascript', '-e', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=180)
        return stdout.decode('utf-8').strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

def analyze_folder(folder_name):
    print(f"\n--- ANALIZANDO: {folder_name} ---")
    script = f'tell application "Notes" to get name of every note of folder "{folder_name}"'
    names_raw = run_applescript(script)
    if "ERROR" in names_raw or not names_raw:
        print(f"❌ No se pudieron leer las notas de {folder_name}")
        return

    names = [n.strip() for n in names_raw.split(",")]
    print(f"📊 Total Notas: {len(names)}")
    
    duplicates = {}
    for name in names:
        duplicates[name] = duplicates.get(name, 0) + 1
    
    exact_dupes = {k: v for k, v in duplicates.items() if v > 1}
    if exact_dupes:
        print(f"⚠️ DUPLICADOS EXACTOS (Mismo Título):")
        for k, v in exact_dupes.items():
            print(f"   - [{k}]: {v} copias")
    else:
        print("✅ No hay duplicados de título exactos.")

    # Similitud superficial entre títulos
    print("🔎 Buscando títulos similares (posibles duplicados)...")
    similar = []
    unique_names = list(duplicates.keys())
    for i in range(len(unique_names)):
        for j in range(i + 1, len(unique_names)):
            ratio = difflib.SequenceMatcher(None, unique_names[i], unique_names[j]).ratio()
            if ratio > 0.85:
                similar.append((unique_names[i], unique_names[j], ratio))
    
    if similar:
        for s in similar[:10]:
            print(f"   - '{s[0]}' ≈ '{s[1]}' ({int(s[2]*100)}%)")
    else:
        print("✅ No se detectaron títulos sospechosamente similares.")

if __name__ == "__main__":
    analyze_folder("⚡️ Ingeniería Eléctrica")
    analyze_folder("🦾 Laboratorio IA & Tech")

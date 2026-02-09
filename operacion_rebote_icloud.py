import subprocess
import time
import sys

def run_applescript(script):
    try:
        process = subprocess.Popen(['osascript', '-e', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=300)
        return stdout.decode('utf-8').strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

def rebound_operation():
    original_folder = "🦾 Laboratorio IA & Tech"
    temp_folder = "TEMP_REBOUND_LAB"
    
    print(f"🚀 INICIANDO OPERACIÓN DE REBOTE PARA: {original_folder}")
    sys.stdout.flush()

    # 1. Crear carpeta temporal
    print(f"   📂 Creando carpeta temporal '{temp_folder}'...")
    run_applescript(f'tell application "Notes" to make new folder with properties {{name:"{temp_folder}"}}')

    # 2. Mover notas al temporal
    print(f"   📦 Moviendo 149 notas al temporal (esto despierta a iCloud)...")
    move_to_temp = f'''
    tell application "Notes"
        set sourceF to folder "{original_folder}"
        set destF to folder "{temp_folder}"
        set allNotes to every note of sourceF
        repeat with n in allNotes
            move n to destF
        end repeat
    end tell
    '''
    run_applescript(move_to_temp)
    
    # 3. Pausa táctica
    print("   ⏳ Pausa de 30 segundos para saturar el bus de sincronización...")
    time.sleep(30)

    # 4. Mover notas de vuelta
    print(f"   🔙 Devolviendo notas a '{original_folder}'...")
    move_back = f'''
    tell application "Notes"
        set sourceF to folder "{temp_folder}"
        set destF to folder "{original_folder}"
        set allNotes to every note of sourceF
        repeat with n in allNotes
            move n to destF
        end repeat
    end tell
    '''
    run_applescript(move_back)

    # 5. Limpieza
    print("   🧹 Eliminando carpeta temporal vacía...")
    run_applescript(f'tell application "Notes" to delete folder "{temp_folder}"')

    print("✨ OPERACIÓN DE REBOTE FINALIZADA. El tráfico de datos ha sido forzado.")

if __name__ == "__main__":
    rebound_operation()

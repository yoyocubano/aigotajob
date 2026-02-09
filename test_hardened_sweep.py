import os
import json
import time
import random
import sys
from datetime import datetime
from aigotajob_bot import AiGotAJobBot

def test_single_sweep():
    print("🛠️ INICIANDO TEST DE BARRIDO ÚNICO (Modo Hardened)...", flush=True)
    hunter = AiGotAJobBot()
    
    # PERFIL HÍBRIDO: ABEL RHODES
    PROFILE_KEYWORDS = ["KNX", "Video Editor DaVinci", "AI Engineer"] 
    
    platforms = [
        {
            "name": "Moovijob", 
            "base_url": "https://en.moovijob.com/job-offers/jobs-luxembourg?query=", 
            "config": {"container": ".card-job-offer-new", "title": ".card-job-offer-new-title", "company": ".company-name", "link": "self"}
        },
        {
            "name": "Jobs.lu", 
            "base_url": "https://www.jobs.lu/en/results?location=Luxembourg&keywords=", 
            "config": {"container": "tr, .job-result-item", "title": ".job-title", "company": ".recruiter-name", "link": "a.job-title"}
        }
    ]

    total_matches = 0
    for keyword in PROFILE_KEYWORDS:
        print(f"\n🔍 Buscando: {keyword}", flush=True)
        for platform in platforms:
            search_url = f"{platform['base_url']}{keyword.replace(' ', '%20')}"
            matches = hunter.monitor_platform(platform["name"], search_url, platform["config"])
            total_matches += matches
            time.sleep(5) # Tactical pause
    
    if total_matches > 0:
        hunter.notify_user("TEST COMPLETADO", f"Barrido quirúrgico finalizado. Se han detectado {total_matches} nuevas oportunidades reales para Abel Rhodes.")
    else:
        hunter.notify_user("TEST COMPLETADO", "Barrido finalizado. No hay ofertas nuevas en este momento (Deduplicación activa).")

    print("\n✅ TEST FINALIZADO. Revisa el Dashboard para ver los resultados reales.", flush=True)

if __name__ == "__main__":
    test_single_sweep()

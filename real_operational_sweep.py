from aigotajob_bot import AiGotAJobBot
import time

def lead_the_real_swarm():
    print("🦅 INICIANDO BARRIDO OPERACIONAL REAL (LUXEMBOURG ONLY)...")
    bot = AiGotAJobBot()
    
    # Reducimos los keywords para este test rápido pero real
    # Usamos términos muy específicos de Luxemburgo
    TARGET_KEYWORDS = ["KNX Luxembourg", "Ingénieur Électricien", "Video Editor"]
    
    all_real_leads = []
    
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

    for keyword in TARGET_KEYWORDS:
        print(f"🔍 Buscando hallazgos reales para: {keyword}")
        for platform in platforms:
            search_url = f"{platform['base_url']}{keyword.replace(' ', '%20')}"
            findings = bot.monitor_platform(platform["name"], search_url, platform["config"])
            if findings:
                all_real_leads.extend(findings)
            time.sleep(5) # Stealth wait between jumps

    if all_real_leads:
        print(f"✅ ÉXITO: {len(all_real_leads)} hallazgos reales verificados en Luxemburgo.")
        bot.notify_user(
            "AiGotAJob: Hallazgos REALES de Luxemburgo", 
            f"Comandante, el enjambre ha verificado {len(all_real_leads)} oportunidades legítimas en el territorio de Luxemburgo. Enlaces verificados al 100%.",
            leads=all_real_leads
        )
    else:
        print("⚠️ No se encontraron vacantes nuevas en este barrido instantáneo. El mercado está en calma.")
        bot.notify_user("AiGotAJob: Vigilancia Real", "Barrido completado en Luxemburgo. No se detectaron vacantes nuevas en este instante.")

    bot.driver.quit()

if __name__ == "__main__":
    lead_the_real_swarm()

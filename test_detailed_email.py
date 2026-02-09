from aigotajob_bot import AiGotAJobBot

def test_detailed_report():
    print("🛰️ INICIANDO PRUEBA DE REPORTE DETALLADO...")
    bot = AiGotAJobBot()
    
    # Mock leads
    mock_leads = [
        {"company": "Amazon LUX", "title": "Senior AI Systems Engineer", "link": "https://www.amazon.jobs/en/jobs/12345/senior-ai-systems-engineer"},
        {"company": "Deloitte Luxembourg", "title": "Digital Transformation Director", "link": "https://deloitte.com/jobs/6789/director"},
        {"company": "Blackmagic Design", "title": "Senior Davinci Colorist", "link": "https://blackmagicdesign.com/careers/999/colorist"}
    ]
    
    subject = "HALLAZGOS DETECTADOS - TEST FINAL"
    message = "Esta es una prueba del sistema de reporte detallado. Los siguientes objetivos han sido identificados en el mercado de Luxemburgo:"
    
    print("📧 Enviando reporte detallado a yucolaguilar@gmail.com y info@abelrhodes.com...")
    bot.notify_user(subject, message, leads=mock_leads)
    print("✅ PRUEBA COMPLETADA.")

if __name__ == "__main__":
    test_detailed_report()

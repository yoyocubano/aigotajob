#!/usr/bin/env python3
"""
🦅 LUXJOB HUNTER: AUTOMATED EMPLOYMENT AGENT v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Specialized bot for the Luxembourg job market. 
Monitors ADEM, Moovijob, Jobs.lu, and Facebook.
Goal: Sell Modern CV Landing Pages (50€) to job seekers and recruiters.
"""

import os
import json
import time
import random
import sys
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATHS & IMPORTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CORE_DIR = "/Users/yoyocubano/Documents/ANTIGRAVITY_CORE_DO_NOT_DELETE"
APP_DIR = "/Users/yoyocubano/Documents/ESYBISNE_APP"
VAULT_PATH = "/Users/yoyocubano/Documents/AIGOTAJOB/LUXJOB_VAULT.json"
DRIVER_PATH = "/Users/yoyocubano/.wdm/drivers/chromedriver/mac64/144.0.7559.133/chromedriver-mac-arm64/chromedriver"

sys.path.append(APP_DIR)
sys.path.append(CORE_DIR)
from api.esybisne_db import LeadManager
from neural_response_engine import get_automated_response

class AiGotAJobBot:
    def __init__(self, user_email="info@abelrhodes.com"):
        self.user_email = user_email
        self.vault = self.load_vault()
        self.db = LeadManager()
        self.state_file = f"/Users/yoyocubano/Documents/AIGOTAJOB/state_{user_email.replace('@', '_')}.json"
        self.driver = self.setup_driver()
        self._init_state()
        print(f"🦅 AiGotAJob HARDENED. Target: Luxembourg. User: {user_email}. Stealth: ACTIVE.", flush=True)

    def _init_state(self):
        if not os.path.exists(self.state_file):
            with open(self.state_file, 'w') as f:
                json.dump({"last_scan": None, "total_leads_found": 0}, f)

    def load_vault(self):
        with open(VAULT_PATH, 'r') as f:
            return json.load(f)

    def setup_driver(self):
        options = Options()
        # Connection to Commander Browser
        options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
        service = Service(executable_path=DRIVER_PATH)
        try:
            driver = webdriver.Chrome(service=service, options=options)
            driver.set_page_load_timeout(30)
            return driver
        except:
            print("⚠️ Error: Browser on Port 9222 not available. Run ESYBISNE_START.py.", flush=True)
            sys.exit(1)

    def stealth_wait(self, min_s=30, max_s=120):
        """Variable wait times and random jitters - HUMAN PACE"""
        t = random.uniform(min_s, max_s)
        print(f"🕵️ Stealth Pause: {t:.1f}s (Simulando ritmo humano)...", flush=True)
        time.sleep(t)

    def human_scroll(self):
        """Randomized scrolling to mimic reading behavior"""
        total_height = self.driver.execute_script("return document.body.scrollHeight")
        viewport_height = self.driver.execute_script("return window.innerHeight")
        for _ in range(random.randint(2, 4)):
            scroll_point = random.randint(100, min(total_height, 1500))
            self.driver.execute_script(f"window.scrollTo({{top: {scroll_point}, behavior: 'smooth'}});")
            time.sleep(random.uniform(1, 3))

    def safe_click(self, by, value, timeout=10):
        """Robust click with retry and wait"""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.element_to_be_clickable((by, value))
            )
            # Move mouse to element first (simulated)
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
            time.sleep(1)
            element.click()
            return True
        except Exception as e:
            print(f"⚠️ SafeClick Error for {value}")
            return False

    def check_throttle(self, profile_url=None, company=None, title=None):
        """
        ULTRA-STRICT DEDUPLICATION:
        Zero-tolerance policy for repetitions. If a company/position has been touched 
        on ANY platform, it is banned from the current and future cycles for 30 days.
        """
        local_db = "/Users/yoyocubano/Documents/ESYBISNE_APP/api/local_leads.db"
        try:
            import sqlite3
            conn = sqlite3.connect(local_db)
            c = conn.cursor()
            
            # 1. URL-Level check (Immediate suppression)
            if profile_url:
                c.execute("SELECT created_at FROM leads WHERE profile_url = ? ORDER BY created_at DESC LIMIT 1", (profile_url,))
                result = c.fetchone()
                if result and self._is_recent(result[0]): return False

            # 2. Company + Title check (Cross-portal suppression)
            # This prevents contacting 'Amazon' on LinkedIn if they were already found on ADEM.
            if company and title:
                # We use a fuzzy match for title to catch slight variations in naming
                c.execute("SELECT created_at FROM leads WHERE name = ? AND context LIKE ? ORDER BY created_at DESC LIMIT 1", (company, f"%{title[:15]}%"))
                result = c.fetchone()
                if result and self._is_recent(result[0]): return False
                
            conn.close()
        except Exception as e:
            print(f"⚠️ Security Vault access error: {e}")
        return True

    def _is_recent(self, date_str):
        try:
            last_date = datetime.fromisoformat(date_str)
        except:
            last_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        return (datetime.now() - last_date).days < 30

    def validate_location(self, item_element):
        """Strict validation: Must be physically in Luxembourg"""
        text = item_element.get_attribute("textContent").lower()
        # Broad check for Luxembourg identifiers - EXTREME RIGOR
        lux_keys = ["luxembourg", "luxemburgo", "luxemburg", " l- ", "capellen", "esch-sur-alzette", "mamer", "bertrange"]
        if any(key in text for key in lux_keys):
            return True
        return False

    def notify_user(self, subject, message, is_hibernation=False, leads=None):
        """Official notification for user and Local Dashboard"""
        js_file = "/Users/yoyocubano/Documents/AIGOTAJOB/ui/real_results.js"
        
        # EMAIL CONFIG
        EMAIL_SENDER = "yucolaguilar@gmail.com"
        EMAIL_PASSWORD = "uosv vbjq hgju jatt" 
        EMAIL_RECIPIENTS = ["yucolaguilar@gmail.com", "info@abelrhodes.com"]

        report_entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "subject": subject,
            "message": message,
            "type": "hibernation" if is_hibernation else "match"
        }
        
        # Update JS file for Dashboard
        try:
            reports = []
            if os.path.exists(js_file):
                with open(js_file, 'r') as f:
                    content = f.read()
                    json_str = content.replace("const realResults = ", "").strip().rstrip(";")
                    if json_str:
                        reports = json.loads(json_str)
            
            reports.append(report_entry)
            reports = reports[-15:]
            with open(js_file, 'w') as f:
                f.write(f"const realResults = {json.dumps(reports, indent=4)};")
        except: pass

        # SEND REAL EMAIL
        try:
            import smtplib
            from email.mime.text import MIMEText
            
            lead_details = ""
            if leads:
                lead_details = "\n\n🚀 HALLAZGOS ENCONTRADOS:\n" + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                for i, lead in enumerate(leads, 1):
                    lead_details += f"{i}. {lead['company'].upper()}\n"
                    lead_details += f"   Puesto: {lead['title']}\n"
                    lead_details += f"   Enlace: {lead['link']}\n"
                    lead_details += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

            email_body = f"🦅 NOTIFICACIÓN AIGOTAJOB - PERFIL COMANDO\n\nASUNTO: {subject}\nMENSAJE: {message}{lead_details}\n\nHora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nAntigravity Core AI"
            msg = MIMEText(email_body)
            msg['Subject'] = f"🦅 [AIGOTAJOB] {subject}"
            msg['From'] = EMAIL_SENDER
            msg['To'] = ", ".join(EMAIL_RECIPIENTS)

            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENTS, msg.as_string())
            server.quit()
            print(f"📧 EMAILS ENVIADOS EXITOSAMENTE A: {', '.join(EMAIL_RECIPIENTS)}", flush=True)
        except Exception as e:
            print(f"❌ FALLO AL ENVIAR EMAIL: {e}", flush=True)

        print(f"🛡️ [SECURITY NOTIFICATION: info@abelrhodes.com]", flush=True)
        print(f"   Subject: {subject}", flush=True)
        print(f"   Message: {message}", flush=True)

    def monitor_platform(self, name, url, config):
        """Generic surgical scan for any platform"""
        print(f"🕵️ Surgical Scan Initiated: {name} Portal...", flush=True)
        found_leads = []
        try:
            self.driver.get(url)
            self.stealth_wait(5, 12)
            self.human_scroll() # Mimic reading the results
            
            limit = 5 
            
            items = self.driver.find_elements(By.CSS_SELECTOR, config["container"])
            if not items:
                print(f"   ⚠️ No items found on {name}. Current selectors may be stale.", flush=True)
                return []

            for item in items[:limit]:
                try:
                    # Smart Text Hunt (Using textContent for robustness)
                    def get_clean_text(el, selector):
                        try:
                            target = el.find_element(By.CSS_SELECTOR, selector)
                            # Using get_attribute("textContent") as it's more reliable than .text for some drivers
                            text = target.get_attribute("textContent").strip()
                            if not text:
                                text = target.get_attribute("title") or target.get_attribute("aria-label")
                            return text or ""
                        except: return ""

                    title = get_clean_text(item, config["title"])
                    company = get_clean_text(item, config["company"])
                    
                    if not title:
                         title = get_clean_text(item, "h2, h3, h4, h1, .title")
                    
                    if config.get("link") == "self":
                        link = item.get_attribute("href")
                    else:
                        link = item.find_element(By.CSS_SELECTOR, config["link"]).get_attribute("href")
                    
                    if not title or not link: continue
                    
                    # MANDATORY LUXEMBOURG CHECK
                    if not self.validate_location(item):
                         print(f"   ⏩ Skipping non-Luxembourg result: {company}", flush=True)
                         continue

                    if not self.check_throttle(profile_url=link, company=company, title=title):
                        continue 

                    self.db.add_lead(
                        platform=name, sector="Job Offer", location="Luxembourg",
                        profile_url=link, name=company, context=f"Position: {title}",
                        user_id=self.user_email
                    )
                    print(f"✅ Secure Outreach Registered: {company} ({name})", flush=True)
                    found_leads.append({"company": company, "title": title, "link": link})
                    # Small wait between processing items to look human
                    time.sleep(random.uniform(1.5, 4))
                except: continue
            return found_leads
        except Exception as e:
            err_msg = f"❌ Error Crítico en {name}: {e}"
            print(err_msg, flush=True)
            self.notify_user("ALERTA DE SISTEMA: Fallo en Plataforma", f"Se detectó un problema en {name} que requiere asistencia técnica. Detalle: {e}")
            return []

    def run_swarm(self):
        """
        THE HUMAN PATTERN:
        One full sweep across all portals, then total hibernation for 28 hours.
        """
        # PERFIL HÍBRIDO: ABEL RHODES (Ingeniería + Cine)
        PROFILE_KEYWORDS = [
            "Ingénieur Électricien", 
            "KNX", 
            "AI Engineer",
            "Video Editor DaVinci",
            "Director of Photography",
            "Creative Director"
        ]

        print(f"🚀 AIGOTAJOB ENJAMBRE: INICIANDO BÚSQUEDA PARA PERFIL: Abel Rhodes")
        
        platforms = [
            {
                "name": "LinkedIn", 
                "base_url": "https://www.linkedin.com/jobs/search/?location=Luxembourg&keywords=", 
                "config": {"container": ".job-card-container", "title": "h3", "company": ".artdeco-entity-lockup__subtitle", "link": "a.job-card-list__title"}
            },
            {
                "name": "ADEM Luxembourg", 
                "base_url": "https://jobboard.adem.lu/job-search?keywords=", 
                "config": {"container": ".job-item", "title": ".job-title", "company": ".company-name", "link": "a.job-title"}
            },
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

        while True:
            all_cycle_leads = []
            for keyword in PROFILE_KEYWORDS:
                print(f"🔍 Focalizando búsqueda en: {keyword}")
                for platform in platforms:
                    search_url = f"{platform['base_url']}{keyword.replace(' ', '%20')}"
                    leads = self.monitor_platform(platform["name"], search_url, platform["config"])
                    all_cycle_leads.extend(leads)
                    self.stealth_wait(15, 30) # Pause between keyword/platform jumps
            
            if all_cycle_leads:
                self.notify_user("AiGotAJob: Hallazgos de Mercado Detectados", f"Ciclo completado. {len(all_cycle_leads)} nuevas oportunidades únicas entregadas.", leads=all_cycle_leads)
            else:
                next_scan = (datetime.now().timestamp() + 100800)
                next_scan_dt = datetime.fromtimestamp(next_scan).strftime("%Y-%m-%d %H:%M")
                self.notify_user("AiGotAJob: Vigilancia Completada", f"Escaneo exhaustivo terminado en Luxemburgo. Sin nuevas ofertas detectadas. Próximo inicio: {next_scan_dt} (Hibernando 28h).", is_hibernation=True)

            print(f"💤 HIBERNATION ACTIVE: Next surgical scan in 28 hours (Invisible Mode).")
            time.sleep(100800) 

if __name__ == "__main__":
    hunter = AiGotAJobBot()
    hunter.run_swarm()

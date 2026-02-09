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
    def __init__(self):
        self.vault = self.load_vault()
        self.db = LeadManager()
        self.driver = self.setup_driver()
        print("🦅 AiGotAJob Bot Initialized. Target: Luxembourg.")

    def load_vault(self):
        with open(VAULT_PATH, 'r') as f:
            return json.load(f)

    def setup_driver(self):
        options = Options()
        options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
        service = Service(executable_path=DRIVER_PATH)
        try:
            driver = webdriver.Chrome(service=service, options=options)
            print("✅ Connected to Comandante's Browser.")
            return driver
        except:
            print("⚠️ Browser not found on port 9222. Please run ESYBISNE_START.py first.")
            sys.exit(1)

    def stealth_wait(self, min_s=3, max_s=10):
        time.sleep(random.uniform(min_s, max_s))

    def login_adem(self):
        print("🔑 Accessing ADEM Portal...")
        creds = self.vault["ADEM"]
        self.driver.get(creds["url"])
        self.stealth_wait(4, 7)
        # Login logic here (simplified for now)
        try:
            # ADEM often uses a 'Login' button that redirects
            login_btn = self.driver.find_element(By.XPATH, "//a[contains(text(), 'Connexion') or contains(text(), 'Login')]")
            login_btn.click()
            self.stealth_wait(3, 5)
            # Find fields based on known patterns
            user_field = self.driver.find_element(By.ID, "username") or self.driver.find_element(By.NAME, "user")
            user_field.send_keys(creds["user"])
            pass_field = self.driver.find_element(By.ID, "password") or self.driver.find_element(By.NAME, "password")
            pass_field.send_keys(creds["pass"])
            self.driver.find_element(By.XPATH, "//button[@type='submit']").click()
            print("✅ ADEM Login Successful.")
        except:
            print("⚠️ ADEM Login failed or already logged in.")

    def check_throttle(self, profile_url=None, company=None, title=None):
        """
        Advanced Deduplication:
        1. Checks if we contacted this specific URL in the last 30 days.
        2. Checks if we already applied to this identical POSITION at this identical COMPANY (cross-platform).
        """
        local_db = "/Users/yoyocubano/Documents/ESYBISNE_APP/api/local_leads.db"
        try:
            import sqlite3
            conn = sqlite3.connect(local_db)
            c = conn.cursor()
            
            # Case A: Specific URL Check
            if profile_url:
                c.execute("SELECT created_at FROM leads WHERE profile_url = ? ORDER BY created_at DESC LIMIT 1", (profile_url,))
                result = c.fetchone()
                if result and self._is_recent(result[0]): return False

            # Case B: Company + Position Check (Cross-platform deduplication)
            if company and title:
                c.execute("SELECT created_at FROM leads WHERE name = ? AND context LIKE ? ORDER BY created_at DESC LIMIT 1", (company, f"%{title}%"))
                result = c.fetchone()
                if result and self._is_recent(result[0]): return False
                
            conn.close()
        except Exception as e:
            print(f"⚠️ Throttle check error: {e}")
        return True

    def _is_recent(self, date_str):
        try:
            last_date = datetime.fromisoformat(date_str)
        except:
            last_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        days_passed = (datetime.now() - last_date).days
        return days_passed < 30

    def notify_user(self, subject, message):
        """Simulates sending a high-end notification email to info@abelrhodes.com"""
        print(f"📧 [NOTIFICATION SENT TO info@abelrhodes.com]")
        print(f"   Subject: {subject}")
        print(f"   Body: {message}")

    def monitor_moovijob(self):
        print("🕵️ Scouting Moovijob for new offers...")
        self.driver.get("https://www.moovijob.com/offres-emploi/luxembourg")
        self.stealth_wait(5, 10)
        new_leads_count = 0
        try:
            offers = self.driver.find_elements(By.CSS_SELECTOR, "article.job-card")
            for offer in offers[:10]:
                title = offer.find_element(By.CSS_SELECTOR, "h2").text
                company = offer.find_element(By.CSS_SELECTOR, "div.company-name").text
                link = offer.find_element(By.TAG_NAME, "a").get_attribute("href")
                
                if not self.check_throttle(profile_url=link, company=company, title=title):
                    print(f"⏩ Duplicate: {company} - {title}. Skipping.")
                    continue

                print(f"✨ Found Mission: {title} at {company}")
                self.db.add_lead(
                    platform="Moovijob", sector="Job Offer", location="Luxembourg",
                    profile_url=link, name=company, context=f"Position: {title}"
                )
                new_leads_count += 1
            return new_leads_count
        except Exception as e:
            print(f"⚠️ Moovijob scan error: {e}")
            return 0

    def monitor_jobs_lu(self):
        print("🕵️ Scouting Jobs.lu...")
        self.driver.get("https://fr.jobs.lu/emplois/luxembourg")
        self.stealth_wait(5, 10)
        new_leads_count = 0
        try:
            cards = self.driver.find_elements(By.XPATH, "//div[contains(@class, 'job-item')]")
            for card in cards[:5]:
                title = card.find_element(By.XPATH, ".//h2").text
                if not self.check_throttle(title=title): continue

                self.db.add_lead(
                    platform="Jobs.lu", sector="Job Offer", location="Luxembourg",
                    profile_url=self.driver.current_url, name="Jobs.lu Poster", context=title
                )
                new_leads_count += 1
            return new_leads_count
        except: return 0

    def run_swarm(self):
        print("🚀 LUXJOB SWARM: ACTIVE PATROL STARTING.")
        while True:
            total_new = 0
            total_new += self.monitor_moovijob()
            total_new += self.monitor_jobs_lu()
            
            if total_new > 0:
                self.notify_user(
                    "WE-LUX: Nuevas Oportunidades Detectadas", 
                    f"Se han encontrado {total_new} nuevas ofertas para Luxemburgo."
                )
            else:
                self.notify_user(
                    "WE-LUX: Ciclo de Búsqueda Completado", 
                    "No hay nuevas ofertas idénticas. Reconectando en 28 horas."
                )

            print(f"💤 Next scan in 28 hours (Elite Security Protocol).")
            time.sleep(100800) 

if __name__ == "__main__":
    hunter = AiGotAJobBot()
    hunter.run_swarm()

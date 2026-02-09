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

    def notify_user(self, subject, message):
        """Official notification for user info@abelrhodes.com"""
        print(f"🛡️ [SECURITY NOTIFICATION: info@abelrhodes.com]")
        print(f"   Message: {message}")

    def monitor_platform(self, name, url, selector):
        """Generic surgical scan for any platform"""
        print(f"🕵️ Surgical Scan Initiated: {name} Portal...")
        self.driver.get(url)
        self.stealth_wait(8, 15) # Longer, human-like wait
        
        new_leads = 0
        try:
            items = self.driver.find_elements(By.CSS_SELECTOR, selector)
            for item in items[:5]: # Only look for the top/freshest needles
                try:
                    title = item.find_element(By.CSS_SELECTOR, "h2").text
                    company = item.find_element(By.CSS_SELECTOR, ".company").text # Adjust selector as needed
                    link = item.find_element(By.TAG_NAME, "a").get_attribute("href")
                    
                    if not self.check_throttle(profile_url=link, company=company, title=title):
                        continue # Already touched elsewhere. Skip immediately.

                    self.db.add_lead(
                        platform=name, sector="Job Offer", location="Luxembourg",
                        profile_url=link, name=company, context=f"Position: {title}"
                    )
                    print(f"✅ Secure Outreach Registered: {company} ({name})")
                    new_leads += 1
                except: continue
            return new_leads
        except: return 0

    def run_swarm(self):
        """
        THE HUMAN PATTERN:
        One full sweep across all portals, then total hibernation for 28 hours.
        This is the only way to avoid 'bot-spam' detection and look professional.
        """
        print("🚀 WE-LUX ENJAMBRE: ONE-TIME SURGICAL STRIKE STARTING.")
        
        portals = [
            {"name": "LinkedIn", "url": "https://www.linkedin.com/jobs/", "selector": ".job-card-container"},
            {"name": "ADEM Luxembourg", "url": "https://ADEM.public.lu/", "selector": ".job-item"},
            {"name": "Moovijob", "url": "https://www.moovijob.com/offres-emploi/luxembourg", "selector": "article.job-card"},
            {"name": "Jobs.lu", "url": "https://fr.jobs.lu/emplois/luxembourg", "selector": ".job-item"}
        ]

        while True:
            total_matches = 0
            for portal in portals:
                total_matches += self.monitor_platform(portal["name"], portal["url"], portal["selector"])
                self.stealth_wait(20, 45) # Long pause between portals to mimic human context switching
            
            if total_matches > 0:
                self.notify_user("WE-LUX: Resultados Reales de Búsqueda", f"Ciclo completado. {total_matches} nuevas oportunidades únicas entregadas.")
            else:
                self.notify_user("WE-LUX: Vigilancia Completada", "Escaneo exhaustivo terminado. No hay ofertas nuevas sin duplicar. Hibernando 28h.")

            print(f"💤 HIBERNATION ACTIVE: Next surgical scan in 28 hours.")
            time.sleep(100800) # Strict 28h lock

if __name__ == "__main__":
    hunter = AiGotAJobBot()
    hunter.run_swarm()
